from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
from modelscope.hub.snapshot_download import snapshot_download
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from layers.Embed import TokenEmbedding
from models.TimeLLM import ReprogrammingLayer


def segment_series(x: torch.Tensor, token_len: int) -> torch.Tensor:
    """[B, L, C] -> [B*C, L/token_len, token_len], without padding."""
    if x.ndim != 3:
        raise ValueError(f"expected [B, L, C], got {tuple(x.shape)}")
    if x.shape[1] % token_len != 0:
        raise ValueError(
            f"seq_len={x.shape[1]} must be divisible by token_len={token_len}"
        )
    batch, seq_len, channels = x.shape
    return (
        x.permute(0, 2, 1)
        .contiguous()
        .view(batch * channels, seq_len // token_len, token_len)
    )


def build_shift_target(
    x_enc: torch.Tensor, future: torch.Tensor, token_len: int
) -> torch.Tensor:
    """Build token-wise next-segment targets on the original data scale."""
    if future.shape[1] < token_len:
        raise ValueError(
            f"future length {future.shape[1]} is smaller than token_len={token_len}"
        )
    return torch.cat([x_enc[:, token_len:, :], future[:, :token_len, :]], dim=1)


def rollout_steps(horizon: int, token_len: int) -> int:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return (horizon + token_len - 1) // token_len


class SegmentMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Model(nn.Module):
    """TimeLLM/AutoTimes hybrid with token-wise autoregressive supervision."""

    def __init__(self, configs) -> None:
        super().__init__()
        if configs.llm_model != "GPT2":
            raise ValueError("TimeLLM_AutoTimes supports GPT2 only")

        self.seq_len = configs.seq_len
        self.token_len = configs.token_len
        self.token_num = self.seq_len // self.token_len
        self.hidden_dim = configs.llm_dim
        self.d_model = configs.d_model
        self.top_k = 5
        self.eps = 1e-6
        self.use_timestamp = bool(configs.use_timestamp)
        self.use_reprogram = bool(configs.use_reprogram)
        self.use_prompt = bool(configs.use_prompt)

        if self.seq_len % self.token_len != 0:
            raise ValueError(
                f"seq_len={self.seq_len} must be divisible by token_len={self.token_len}"
            )
        if self.hidden_dim != 768:
            raise ValueError("GPT2 hidden dimension must be 768")

        local_model_path = snapshot_download("AI-ModelScope/gpt2")
        self.model_revision = Path(local_model_path).name
        gpt2_config = GPT2Config.from_pretrained(local_model_path)
        gpt2_config.num_hidden_layers = configs.llm_layers
        gpt2_config.output_attentions = False
        gpt2_config.output_hidden_states = False
        self.gpt2 = GPT2Model.from_pretrained(local_model_path, config=gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for parameter in self.gpt2.parameters():
            parameter.requires_grad = False

        self.segment_encoder = SegmentMLP(
            self.token_len,
            self.hidden_dim,
            hidden_dim=configs.mlp_hidden_dim,
            dropout=configs.dropout,
        )
        self.segment_decoder = SegmentMLP(
            self.hidden_dim,
            self.token_len,
            hidden_dim=configs.mlp_hidden_dim,
            dropout=configs.dropout,
        )

        if self.use_timestamp:
            table, metadata = self._load_timestamp_cache(configs.timestamp_cache)
            self._validate_timestamp_metadata(
                metadata, configs, table.shape[0], self.model_revision
            )
            self.register_buffer("timestamp_table", table.float(), persistent=False)
            self.timestamp_scale = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer("timestamp_table", torch.empty(0), persistent=False)

        if self.use_reprogram:
            self.patch_embedding = TokenEmbedding(self.token_len, self.d_model)
            self.word_embeddings = self.gpt2.get_input_embeddings().weight
            word_embeddings = self.word_embeddings
            self.mapping_layer = nn.Linear(word_embeddings.shape[0], 1000)
            self.reprogramming_layer = ReprogrammingLayer(
                self.d_model,
                configs.n_heads,
                configs.d_ff,
                self.hidden_dim,
            )
            self.reprogram_norm = nn.LayerNorm(self.hidden_dim)
            self.fusion_norm = nn.LayerNorm(self.hidden_dim)
            self.reprogram_alpha = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def _load_timestamp_cache(cache_path: str):
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"timestamp cache not found: {path}; run preprocess_gpt2_timestamps.py"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise ValueError(f"invalid timestamp cache format: {path}")
        return payload["embeddings"], payload.get("metadata", {})

    @staticmethod
    def _validate_timestamp_metadata(
        metadata: dict, configs, table_rows: int, model_revision: str
    ) -> None:
        expected = {
            "model_id": "AI-ModelScope/gpt2",
            "model_revision": model_revision,
            "token_len": configs.token_len,
            "hidden_dim": configs.llm_dim,
            "data_path": Path(configs.data_path).name,
            "num_rows": table_rows,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"timestamp cache metadata mismatch: {mismatches}")
        if not metadata.get("frequency"):
            raise ValueError("timestamp cache metadata is missing frequency")

    def _normalize_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _timestamp_embeddings(
        self, time_indices: torch.Tensor, channels: int
    ) -> torch.Tensor:
        if time_indices is None:
            raise ValueError("time indices are required when timestamp embedding is enabled")
        if time_indices.shape[1] != self.token_num:
            raise ValueError(
                f"expected {self.token_num} timestamp tokens, got {time_indices.shape[1]}"
            )
        if time_indices.min() < 0 or time_indices.max() >= self.timestamp_table.shape[0]:
            raise IndexError("timestamp index is outside the precomputed cache")
        embeddings = self.timestamp_table[time_indices.long()]
        if channels > 1:
            embeddings = embeddings.repeat_interleave(channels, dim=0)
        return embeddings

    def _prompt_embeddings(self, x_enc: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x_enc.shape
        flat = x_enc.permute(0, 2, 1).contiguous().view(batch * channels, seq_len, 1)
        minimum = flat.min(dim=1).values[:, 0]
        maximum = flat.max(dim=1).values[:, 0]
        median = flat.median(dim=1).values[:, 0]
        trend = flat.diff(dim=1).sum(dim=1)[:, 0]
        fft = torch.fft.rfft(flat.permute(0, 2, 1), dim=-1)
        corr = torch.fft.irfft(fft * torch.conj(fft), dim=-1).mean(dim=1)
        lags = torch.topk(corr, self.top_k, dim=-1).indices

        prompts = []
        for index in range(flat.shape[0]):
            prompts.append(
                "<|start_prompt|>Dataset description: The Electricity Transformer "
                "Temperature (ETT) is a crucial indicator in electric power deployment. "
                f"Task description: forecast the next {self.token_len} steps given the "
                f"previous {self.seq_len} steps; Input statistics: min value "
                f"{minimum[index].item()}, max value {maximum[index].item()}, median value "
                f"{median[index].item()}, trend is "
                f"{'upward' if trend[index] > 0 else 'downward'}, top 5 lags are "
                f"{lags[index].tolist()}<|end_prompt|>"
            )
        token_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids.to(x_enc.device)
        return self.gpt2.get_input_embeddings()(token_ids)

    def _encode(
        self,
        x_normalized: torch.Tensor,
        x_raw: torch.Tensor,
        time_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, channels = x_normalized.shape
        segments = segment_series(x_normalized, self.token_len)
        direct = self.segment_encoder(segments)

        if self.use_timestamp:
            timestamps = self._timestamp_embeddings(time_indices, channels)
            direct = self._normalize_embedding(direct) + self.timestamp_scale * self._normalize_embedding(
                timestamps
            )

        if self.use_reprogram:
            patch_tokens = self.patch_embedding(segments)
            source = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            reprogrammed = self.reprogramming_layer(patch_tokens, source, source)
            gate = torch.sigmoid(self.reprogram_alpha)
            direct = self.fusion_norm(direct + gate * self.reprogram_norm(reprogrammed))

        if self.use_prompt:
            prompt = self._prompt_embeddings(x_raw)
            llm_input = torch.cat([prompt, direct], dim=1)
        else:
            llm_input = direct
        hidden = self.gpt2(inputs_embeds=llm_input).last_hidden_state
        return hidden[:, -self.token_num :, :]

    def forward(
        self, x_enc: torch.Tensor, time_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        normalized = centered / stdev

        batch, _, channels = x_enc.shape
        hidden = self._encode(normalized, x_enc, time_indices)
        decoded = self.segment_decoder(hidden)
        decoded = decoded.view(batch, channels, self.seq_len).permute(0, 2, 1).contiguous()
        return decoded, means, stdev

    def predict_next(
        self, x_enc: torch.Tensor, time_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        decoded, means, stdev = self.forward(x_enc, time_indices)
        return decoded[:, -self.token_len :, :] * stdev + means
