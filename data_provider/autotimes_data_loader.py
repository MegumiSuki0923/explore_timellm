import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Dataset_ETTh1_AutoTimes(Dataset):
    """ETTh1 loader with channel-independent samples and segment timestamp indices."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str,
        seq_len: int,
        pred_len: int,
        token_len: int,
        features: str = "M",
        target: str = "OT",
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"unsupported flag: {flag}")
        if seq_len % token_len != 0:
            raise ValueError("seq_len must be divisible by token_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.token_len = token_len
        self.token_num = seq_len // token_len
        self.future_token_num = math.ceil(pred_len / token_len)
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        frame = pd.read_csv(os.path.join(root_path, data_path))
        border1s = [
            0,
            12 * 30 * 24 - seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        if features in {"M", "MS"}:
            data_frame = frame[frame.columns[1:]]
        elif features == "S":
            data_frame = frame[[target]]
        else:
            raise ValueError(f"unsupported features mode: {features}")

        self.scaler = StandardScaler()
        self.scaler.fit(data_frame.iloc[border1s[0] : border2s[0]].values)
        scaled = self.scaler.transform(data_frame.values)
        self.data_x = scaled[self.border1 : self.border2]
        self.data_y = self.data_x
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - seq_len - pred_len + 1
        if self.tot_len <= 0:
            raise ValueError("dataset split is shorter than seq_len + pred_len")

    def __getitem__(self, index: int):
        feature_id = index // self.tot_len
        start = index % self.tot_len
        end = start + self.seq_len
        target_end = end + self.pred_len

        seq_x = self.data_x[start:end, feature_id : feature_id + 1]
        seq_y = self.data_y[end:target_end, feature_id : feature_id + 1]
        global_start = self.border1 + start
        x_time_indices = global_start + np.arange(self.token_num) * self.token_len
        y_time_indices = (
            global_start
            + self.seq_len
            + np.arange(self.future_token_num) * self.token_len
        )
        return seq_x, seq_y, x_time_indices.astype(np.int64), y_time_indices.astype(np.int64)

    def __len__(self) -> int:
        return self.tot_len * self.enc_in


def autotimes_data_provider(args, flag: str, pred_len: int):
    if args.data != "ETTh1":
        raise ValueError("the first TimeLLM_AutoTimes implementation supports ETTh1 only")
    dataset = Dataset_ETTh1_AutoTimes(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=pred_len,
        token_len=args.token_len,
        features=args.features,
        target=args.target,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag == "train",
        num_workers=args.num_workers,
        drop_last=True,
    )
    return dataset, loader
