from math import sqrt, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.hub.snapshot_download import snapshot_download

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer
)
import transformers
from layers.StandardNorm import Normalize
from layers.Embed import PatchEmbedding
from layers.PeriodDecoder import Decoder, decoder_PredictHead

transformers.logging.set_verbosity_error()


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.period_len = getattr(configs, 'period_len', 24)
        self.llm_chunk_size = configs.llm_chunk_size
        if self.llm_chunk_size < 1:
            raise ValueError('llm_chunk_size must be a positive integer')

        # SparseTSF Segment & Padding parameters
        self.seg_num_x = ceil(self.seq_len / self.period_len)
        self.seg_num_y = ceil(self.pred_len / self.period_len)
        self.pad_len = self.seg_num_x * self.period_len - self.seq_len

        # SparseTSF 1D Convolution temporal aggregation filter
        kernel_size = 1 + 2 * (self.period_len // 2)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        )

        # SparseTSF direct periodic linear forecasting branch
        self.sparse_linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

        # LLM Initialization
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            import os
            local_model_path = '/home/Lain/.cache/modelscope/hub/models/AI-ModelScope/gpt2'
            if not os.path.exists(local_model_path):
                try:
                    local_model_path = snapshot_download('AI-ModelScope/gpt2')
                except Exception:
                    local_model_path = 'gpt2'
            self.gpt2_config = GPT2Config.from_pretrained(local_model_path)
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                config=self.gpt2_config,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
            )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # Full-Resolution Patch Embedding (Uncompressed temporal patches)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_keys=self.d_ff,
            d_llm=self.d_llm
        )

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # LightGTS (ICML 2025) period-parallel decoding: replicate the last
            # token with exponential decay, decode future period tokens, then
            # emit one period block per token (replaces the flatten head).
            self.target_patch_len = 48  # LightGTS ETTh1 finetune recipe: each block emits 48 points (2 periods)
            self.out_patch_num = ceil(self.pred_len / self.target_patch_len)
            decoder_d_ff = 2 * self.d_ff  # LightGTS 2:1 d_model:d_ff convention
            self.decoder = Decoder(
                d_layers=3, patch_len=self.patch_len, d_model=self.d_ff,
                n_heads=configs.n_heads, d_ff=decoder_d_ff,
                attn_dropout=0.4, dropout=0., norm="LayerNorm"
            )
            self.head = decoder_PredictHead(self.d_ff, self.target_patch_len, dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, T, N)
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.shape

        # Permute to Channel-Independent representation: (B * N, T, 1)
        x_ci = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # -------------------------------------------------------------
        # 1. SparseTSF 1D Convolution Temporal Aggregation Filter
        # -------------------------------------------------------------
        if self.pad_len > 0:
            x_pad = F.pad(x_ci.permute(0, 2, 1), (self.pad_len, 0), mode='replicate').permute(0, 2, 1)
        else:
            x_pad = x_ci

        T_pad = x_pad.shape[1]
        x_conv = self.conv1d(x_pad.permute(0, 2, 1)).permute(0, 2, 1) + x_pad  # (B * N, T_pad, 1)

        # -------------------------------------------------------------
        # 2. SparseTSF Direct Periodic Linear Forecasting Branch
        # -------------------------------------------------------------
        x_sparse = x_conv.view(B * N, self.seg_num_x, self.period_len).permute(0, 2, 1)  # (B * N, P, seg_num_x)
        y_sparse = self.sparse_linear(x_sparse).permute(0, 2, 1).contiguous()              # (B * N, seg_num_y, P)
        y_sparse = y_sparse.view(B * N, self.seg_num_y * self.period_len)[:, :self.pred_len]  # (B * N, pred_len)

        # -------------------------------------------------------------
        # 3. Time-LLM High-Resolution Residual Branch
        # -------------------------------------------------------------
        # Extract the smoothed sequence of length T
        x_smooth = x_conv[:, -T:, :]  # (B * N, T, 1)

        # Compute sample-wise statistics
        min_values = torch.min(x_smooth, dim=1)[0]
        max_values = torch.max(x_smooth, dim=1)[0]
        medians = torch.median(x_smooth, dim=1).values
        lags = self.calcute_lags(x_smooth)
        trends = x_smooth.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(B * N):
            min_values_str = f"{min_values[b].item():.2f}"
            max_values_str = f"{max_values[b].item():.2f}"
            median_values_str = f"{medians[b].item():.2f}"
            lags_values_str = str(lags[b].tolist())
            trend_str = "upward" if trends[b].item() > 0 else "downward"
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description} "
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information with period {self.period_len}; "
                f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, "
                f"the trend of input is {trend_str}, dominant periodic lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))  # (B * N, L_p, d_llm)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        embed_dtype = next(self.patch_embedding.parameters()).dtype
        enc_out, _ = self.patch_embedding(x_smooth.permute(0, 2, 1).to(embed_dtype))  # (B * N, patch_nums, d_model)
        source_embeddings = source_embeddings.to(embed_dtype)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # (B * N, patch_nums, d_llm)

        prompt_embeddings = prompt_embeddings.to(enc_out.dtype)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)  # (B * N, L_p + patch_nums, d_llm)

        # Micro-chunking LLM forward pass to keep peak VRAM under control
        chunk_size = self.llm_chunk_size
        total_bn = llama_enc_out.shape[0]
        if total_bn <= chunk_size:
            dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
            dec_out = dec_out[:, -self.patch_nums:, :self.d_ff]
        else:
            dec_out_list = []
            for i in range(0, total_bn, chunk_size):
                chunk_in = llama_enc_out[i:i + chunk_size]
                chunk_out = self.llm_model(inputs_embeds=chunk_in).last_hidden_state
                dec_out_list.append(chunk_out[:, -self.patch_nums:, :self.d_ff])
            dec_out = torch.cat(dec_out_list, dim=0)

        # LightGTS period-parallel decoding: [bs, n_vars, num_patch, d_model]
        head_dtype = next(self.head.parameters()).dtype
        dec_out = dec_out.to(head_dtype).view(B, N, self.patch_nums, self.d_ff)
        dec_out = self.decoder_predict(B, N, dec_out)   # (B, N, d_ff, out_patch_num)
        y_llm = self.head(dec_out, self.target_patch_len)  # (B, out_patch_num * target_patch_len, N)
        y_llm = y_llm[:, :self.pred_len, :].permute(0, 2, 1).reshape(B * N, self.pred_len)  # (B * N, pred_len)

        # -------------------------------------------------------------
        # 4. Residual Fusion & Output Reconstruction
        # -------------------------------------------------------------
        y_total = y_sparse + y_llm  # (B * N, pred_len)
        dec_out = y_total.view(B, N, self.pred_len).permute(0, 2, 1).contiguous()  # (B, pred_len, N)
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def get_dynamic_weights(self, n_preds, decay_rate=0.5):
        """
        Generate dynamic weights for the replicated tokens using an exponential decay scheme.
        (LightGTS_pretrain_period.py:98-111)
        """
        weights = decay_rate ** torch.arange(n_preds)
        return weights

    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        (LightGTS_pretrain_period.py:113-132)
        """
        dec_in = dec_cross[:, :, -1, :].unsqueeze(2).expand(-1, -1, self.out_patch_num, -1)
        weights = self.get_dynamic_weights(self.out_patch_num).to(dec_in.device)
        dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2, 3)

        return decoder_output

    def calcute_lags(self, x_enc):
        # x_enc: (B * N, T, 1) -> permute to (B * N, 1, T)
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)  # (B * N, T)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
