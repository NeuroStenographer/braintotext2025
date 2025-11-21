from __future__ import annotations
import os, h5py, torch
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.utils.rnn as rnn_utils

class BrainDataset(Dataset):
    """
    Reads trials from a single HDF5 split file (data_{train,val,test}.hdf5).
    input_key='input_features', target_key='seq_class_ids' by default.
    """
    def __init__(self, hdf5_file: str, input_key="input_features", target_key="seq_class_ids", is_test=False, use_augmentation=False):
        self.file_path = hdf5_file
        self.input_key = input_key
        self.target_key = target_key
        self.is_test = is_test
        self.use_augmentation = use_augmentation
        self.file = None
        try:
            with h5py.File(self.file_path, "r") as f:
                self.trial_keys = sorted(list(f.keys()))
        except FileNotFoundError:
            self.trial_keys = []

    def __len__(self) -> int:
        return len(self.trial_keys)

    def temporal_mask(self, x: torch.Tensor, pct=0.1, mask_value=0.0) -> torch.Tensor:
        if pct <= 0.0: return x
        seq_len = x.size(0)
        nmask = int(seq_len * pct)
        if nmask > 0:
            idx = torch.randperm(seq_len)[:nmask]
            x[idx, :] = mask_value
        return x

    def __getitem__(self, idx: int):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        k = self.trial_keys[idx]
        g = self.file[k]
        x = torch.tensor(g[self.input_key][:], dtype=torch.float32)
        if self.use_augmentation and not self.is_test:
            x = self.temporal_mask(x, pct=0.1)
        if self.target_key in g:
            y = torch.tensor(g[self.target_key][:], dtype=torch.long)
        else:
            y = torch.tensor([], dtype=torch.long)
        return (x, y, k) if self.is_test else (x, y)

def discover_session_dirs(data_dir: str) -> List[str]:
    return sorted([p.path for p in os.scandir(data_dir) if p.is_dir()])

def build_datasets(data_dir: str) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
    train_sets, val_sets, test_sets = [], [], []
    for sess in discover_session_dirs(data_dir):
        tr = os.path.join(sess, "data_train.hdf5")
        va = os.path.join(sess, "data_val.hdf5")
        te = os.path.join(sess, "data_test.hdf5")
        tr_ds = BrainDataset(tr, is_test=False, use_augmentation=True)
        va_ds = BrainDataset(va, is_test=False, use_augmentation=False)
        te_ds = BrainDataset(te, is_test=True,  use_augmentation=False)
        if len(tr_ds) > 0: train_sets.append(tr_ds)
        if len(va_ds) > 0: val_sets.append(va_ds)
        if len(te_ds) > 0: test_sets.append(te_ds)
    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets)

def collate_ctc(batch):
    is_test = len(batch[0]) == 3
    if is_test:
        xs, ys, keys = zip(*batch)
    else:
        xs, ys = zip(*batch)
    x_lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    y_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    padded_xs = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    padded_ys = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)
    if is_test:
        return padded_xs, padded_ys, x_lengths, y_lengths, keys
    return padded_xs, padded_ys, x_lengths, y_lengths

def build_dataloaders(train_dataset, val_dataset, test_dataset, params_loader: dict):
    bs  = int(params_loader.get("batch_size", 32))
    nw  = int(params_loader.get("num_workers", 0))
    pin = bool(params_loader.get("pin_memory", False))
    shuf= bool(params_loader.get("shuffle_train", True))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=shuf,  num_workers=nw, pin_memory=pin, collate_fn=collate_ctc)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, collate_fn=collate_ctc)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, collate_fn=collate_ctc)
    return train_loader, val_loader, test_loader
