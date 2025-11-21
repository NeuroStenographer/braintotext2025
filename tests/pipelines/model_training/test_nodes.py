import torch
from torch.utils.data import DataLoader, TensorDataset

from braintotext2025.pipelines.model_training.nodes import (
    RecurrentModel,
    TransformerEncModel,
    train_one_epoch,
    validate_one_epoch,
    greedy_decoder,
    run_experiments,
)

def _tiny_vocabulary():
    # super small vocab for quick tests
    vocab = ["AA", "B", "|"]
    blank_id = 0
    token_map = {blank_id: ""} | {i+1: p for i, p in enumerate(vocab)}
    return vocab, token_map, blank_id

def test_recurrent_model_forward():
    vocab, token_map, blank_id = _tiny_vocabulary()
    out_size = len(vocab) + 1  # classes

    model = RecurrentModel(
        model_type="LSTM",
        data_input_size=8,
        adapter_output_size=4,
        hidden_size=16,
        output_size=out_size,
        num_layers=1,
        bidirectional=False,
    )
    x = torch.randn(2, 5, 8)  # [B,S,F]
    y = model(x)
    assert y.shape == (2, 5, out_size)
    # log-softmax along last dim
    assert torch.allclose(y.logsumexp(dim=-1), torch.zeros_like(y[..., 0]))

def test_transformer_model_forward():
    vocab, token_map, blank_id = _tiny_vocabulary()
    out_size = len(vocab) + 1

    model = TransformerEncModel(
        data_input_size=8,
        adapter_output_size=4,
        n_head=2,
        num_layers=1,
        dim_feedforward=16,
        output_size=out_size,
    )
    x = torch.randn(2, 5, 8)
    y = model(x)
    assert y.shape == (2, 5, out_size)

def test_greedy_decoder_basic():
    vocab, token_map, blank_id = _tiny_vocabulary()
    # logits for a single sequence: B=1, S=4, C=|vocab|+1
    C = len(vocab) + 1
    logits = torch.zeros(4, C)
    # force a known path: 1,1,2,0 -> "AA AA B"
    logits[0, 1] = 10
    logits[1, 1] = 10
    logits[2, 2] = 10
    logits[3, 0] = 10
    text = greedy_decoder(logits, token_map, blank_id)
    assert text == "AA B"

def test_train_and_validate_one_epoch_smoke():
    vocab, token_map, blank_id = _tiny_vocabulary()
    out_size = len(vocab) + 1

    model = RecurrentModel(
        model_type="GRU",
        data_input_size=4,
        adapter_output_size=4,
        hidden_size=8,
        output_size=out_size,
        num_layers=1,
        bidirectional=False,
    )
    device = torch.device("cpu")
    model.to(device)

    # make a tiny dataset: 3 sequences, all same length for simplicity
    B, S, F = 3, 5, 4
    x = torch.randn(B, S, F)
    y = torch.randint(1, out_size, (B, S))
    x_len = torch.full((B,), S, dtype=torch.long)
    y_len = torch.full((B,), S, dtype=torch.long)

    ds = TensorDataset(x, y, x_len, y_len)
    loader = DataLoader(ds, batch_size=2)

    crit = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = train_one_epoch(1, model, loader, crit, opt, device)
    val_loss, per = validate_one_epoch(1, model, loader, crit, token_map, blank_id, device)

    assert train_loss >= 0.0
    assert val_loss >= 0.0
    assert 0.0 <= per <= 1.0

def test_run_experiments_minimal(monkeypatch):
    # use a tiny fake loader to keep things fast
    vocab, token_map, blank_id = _tiny_vocabulary()

    B, S, F = 2, 4, 4
    x = torch.randn(B, S, F)
    y = torch.randint(1, len(vocab) + 1, (B, S))
    x_len = torch.full((B,), S, dtype=torch.long)
    y_len = torch.full((B,), S, dtype=torch.long)
    ds = TensorDataset(x, y, x_len, y_len)
    loader = DataLoader(ds, batch_size=2)

    # simple params to avoid touching real files
    params_paths = {
        "args_path": "dummy.yaml",          # will be monkeypatched
        "checkpoint_path": "dummy_ckpt",    # also monkeypatched
    }
    # tiny args.yaml content
    fake_args = {
        "input_size": 4,
        "hidden_size": 8,
        "output_size": len(vocab) + 1,
        "num_layers": 1,
    }

    import yaml
    import io

    def fake_safe_load(f):
        return fake_args

    # monkeypatch yaml.safe_load so run_experiments doesn't need real file
    monkeypatch.setattr(yaml, "safe_load", lambda f: fake_args)

    params_train = {
        "device": "cpu",
        "epochs": 1,
        "lr": 0.001,
        "blank_id": blank_id,
        "vocab": vocab,
        "data_input_size": 4,
        "adapter_output_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "bidirectional": False,
        "n_head": 2,
        "dim_feedforward": 8,
        "use_checkpoint_variants": [
            ["RNN", False],
        ],
    }

    metrics, results_table, submission = run_experiments(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        params_paths=params_paths,
        params_train=params_train,
    )

    # metrics is a dict of experiments
    assert isinstance(metrics, dict)
    assert not results_table.empty
    assert list(results_table.columns)  # has some columns
    # submission df might be empty in minimal case, but should still be a DataFrame
    assert hasattr(submission, "to_csv")
