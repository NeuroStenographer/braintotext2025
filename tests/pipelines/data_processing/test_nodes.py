import os
import h5py
import numpy as np
import torch

from braintotext2025.pipelines.data_processing.nodes import (
    BrainDataset,
    collate_ctc,
    build_datasets,
    build_dataloaders,
)

def _make_dummy_hdf5(path: str, n_trials: int = 3, seq_len: int = 10, n_feat: int = 5):
    """Create a tiny HDF5 file with the same structure as your competition data."""
    with h5py.File(path, "w") as f:
        for i in range(n_trials):
            g = f.create_group(f"trial_{i:04d}")
            g.create_dataset("input_features", data=np.random.randn(seq_len, n_feat).astype("float32"))
            g.create_dataset("seq_class_ids", data=np.random.randint(1, 10, size=(seq_len,), dtype="int32"))
            g.attrs["n_time_steps"] = seq_len
            g.attrs["seq_len"] = seq_len
            g.attrs["session"] = 1
            g.attrs["block_num"] = 1
            g.attrs["trial_num"] = i

def test_brain_dataset_len_and_item(tmp_path):
    h5_path = tmp_path / "data_train.hdf5"
    _make_dummy_hdf5(str(h5_path), n_trials=2)

    ds = BrainDataset(str(h5_path), input_key="input_features", target_key="seq_class_ids", is_test=False)
    assert len(ds) == 2

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    # (seq_len, n_features)
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]

def test_collate_ctc_train_batch():
    # create 3 variable-length sequences
    x1 = torch.randn(5, 4)
    x2 = torch.randn(3, 4)
    x3 = torch.randn(7, 4)
    y1 = torch.randint(1, 5, (5,))
    y2 = torch.randint(1, 5, (3,))
    y3 = torch.randint(1, 5, (7,))
    batch = [(x1, y1), (x2, y2), (x3, y3)]

    padded_x, padded_y, x_len, y_len = collate_ctc(batch)

    # batch_first padding
    assert padded_x.shape[0] == 3
    assert padded_x.shape[1] == 7   # max seq len
    assert padded_x.shape[2] == 4

    assert torch.equal(x_len, torch.tensor([5, 3, 7]))
    assert torch.equal(y_len, torch.tensor([5, 3, 7]))
    assert padded_y.shape[0] == 3
    assert padded_y.shape[1] == 7

def test_build_datasets_scans_session_dirs(tmp_path):
    # simulate session folders each with a train/val/test file
    data_dir = tmp_path / "hdf5_data_final"
    data_dir.mkdir()

    for sess in ["t15.2025.01.10", "t15.2025.03.14"]:
        sess_dir = data_dir / sess
        sess_dir.mkdir()
        for split in ["train", "val", "test"]:
            _make_dummy_hdf5(str(sess_dir / f"data_{split}.hdf5"), n_trials=1)

    train_ds, val_ds, test_ds = build_datasets(str(data_dir))

    # 2 sessions * 1 trial each
    assert len(train_ds) == 2
    assert len(val_ds) == 2
    assert len(test_ds) == 2

def test_build_dataloaders_shapes(tmp_path):
    # reuse build_datasets to create small datasets
    data_dir = tmp_path / "hdf5"
    data_dir.mkdir()
    sess_dir = data_dir / "session_1"
    sess_dir.mkdir()
    _make_dummy_hdf5(str(sess_dir / "data_train.hdf5"), n_trials=4, seq_len=8, n_feat=3)
    _make_dummy_hdf5(str(sess_dir / "data_val.hdf5"),   n_trials=2, seq_len=6, n_feat=3)
    _make_dummy_hdf5(str(sess_dir / "data_test.hdf5"),  n_trials=2, seq_len=5, n_feat=3)

    train_ds, val_ds, test_ds = build_datasets(str(data_dir))
    loader_params = {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle_train": False,
    }
    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, loader_params)

    xb, yb, xlen, ylen = next(iter(train_loader))
    assert xb.shape[0] == 2      # batch size
    assert xb.ndim == 3          # [B, S, F]
    assert xlen.shape[0] == 2
    assert ylen.shape[0] == 2

    # test loader yields test batch with keys in collate_ctc (3-tuple case)
    xb_t, yb_t, xlen_t, ylen_t, keys = next(iter(test_loader))
    assert len(keys) == 2
