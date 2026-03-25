from dataclasses import dataclass
import pandas as pd
import numpy as np
import os

import cv2
import torch
from torch.utils.data import Dataset


ACTIONS = ["LEFT","RIGHT","UP","DOWN","JUMP","ATTACK","HEAL"]


@dataclass
class Batch:
    obs: torch.Tensor
    act: torch.Tensor
    mask_padding: torch.Tensor

class WorldModelDataset(Dataset):
    def __init__(self, dataset_path, context=4, encode=True):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(os.path.join(self.dataset_path, 'aggregated_dataset.csv'))
        self.context = context
        self.encode = encode
        self.action_cols = ACTIONS
        self.num_buttons = len(self.action_cols)

        self.df[self.action_cols] = self.df[self.action_cols]

        self.valid_indices = self._build_indices()

    def _build_indices(self):
        valid = []
        for i in range(len(self.df) - self.context):
            window = self.df.iloc[i:i+self.context+1]

            if len(window["dataset_id"].unique()) != 1:
                continue

            frame_ids = window["frame_id"].values
            if not all(frame_ids[j] + 1 == frame_ids[j+1] for j in range(len(frame_ids)-1)):
                continue

            valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        window = self.df.iloc[i:i+self.context+1]

        frames = []
        actions = []

        for j in range(self.context):
            row = window.iloc[j]

            img = cv2.imread(os.path.join(self.dataset_path, row["frame_low_res_path"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img).permute(2,0,1).float() / 255
            img = img * 2 - 1
            frames.append(img)

            act_row = row[self.action_cols].to_numpy(dtype=np.float32)
            act_encoded = 0
            if self.encode:
                mult = 1
                for k in range(len(act_row)):
                    if act_row[k] > 0:
                        act_encoded += mult
                    mult *= 2
                act = torch.tensor(act_encoded, dtype=torch.long)
            else:
                act = torch.tensor(act_row, dtype=torch.float32)
            actions.append(act)

        frames = torch.stack(frames)
        actions = torch.stack(actions)

        mask = torch.ones(self.context, dtype=torch.bool)

        return {
            'obs': frames,
            'act': actions,
            'mask_padding': mask
        }
