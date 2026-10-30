"""End-to-end run of every command on a small synthetic corpus (CPU, no data)."""

import json

import numpy as np
import pytest
import torch

from ldc.cli import main as ldc
from ldc.data import TE_CLASS_NAMES


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """Normal + 2 fault classes, 312 features, faults shifted on a few sensors."""
    root = tmp_path_factory.mktemp("te")
    rng = np.random.default_rng(0)
    for split, n in (("train", 120), ("test", 80)):
        (root / split).mkdir()
        for cls, name in enumerate(TE_CLASS_NAMES[:3]):
            X = rng.normal(size=(n, 312)).astype(np.float32)
            if cls:
                X[:, cls::52] += 1.5
            ids = np.arange(n) + 10000 * cls
            rows = np.column_stack([ids, X, np.full(n, cls), np.full(n, int(cls > 0))])
            np.save(root / split / f"{name}.npy", rows)
    return root


def _ssl(data_root, out, *extra):
    ldc(["train-ssl", "--data-root", str(data_root), "--out", str(out), "--device", "cpu",
         "--epochs", "2", "--save-epochs", "1,2", *extra])


@pytest.fixture(scope="module")
def init(data_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("sup")
    ldc(["train-sup", "--data-root", str(data_root), "--out", str(out), "--device", "cpu", "--epochs", "2"])
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["train_history"]) == 2
    return out / "encoder_init.pth"


@pytest.mark.parametrize("path", ["full", "detach_h", "freeze_enc", "on_h"])
def test_paths_train_and_diagnose(data_root, init, tmp_path, path):
    run = tmp_path / path
    _ssl(data_root, run, "--init", str(init), "--path", path)
    assert (run / "epoch_2.pth").exists()
    ldc(["diagnose", "--data-root", str(data_root), "--run", str(run), "--init", str(init), "--device", "cpu"])
    diag = json.loads((run / "diagnostics.json").read_text())
    assert [t["epoch"] for t in diag["trajectory"]] == [1, 2]
    assert ("z" in diag["trajectory"][-1]) == (path != "on_h")


def test_freeze_enc_keeps_the_encoder(data_root, init, tmp_path):
    _ssl(data_root, tmp_path / "frozen", "--init", str(init), "--path", "freeze_enc")
    before = torch.load(init, weights_only=False)["context_encoder"]
    after = torch.load(tmp_path / "frozen" / "epoch_2.pth", weights_only=False)["context_encoder"]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_detach_h_blocks_sigreg_only(data_root, init, tmp_path):
    """With α = 0, detach_h leaves nothing to train the encoder."""
    _ssl(data_root, tmp_path / "d", "--init", str(init), "--path", "detach_h", "--alpha", "0")
    before = torch.load(init, weights_only=False)["context_encoder"]
    after = torch.load(tmp_path / "d" / "epoch_2.pth", weights_only=False)["context_encoder"]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_scratch_shared_core_and_report(data_root, init, tmp_path):
    runs = tmp_path / "runs"
    _ssl(data_root, runs / "scratch" / "share0")
    _ssl(data_root, runs / "scratch" / "share25", "--shared-core", "0.25")
    for share in ("share0", "share25"):
        ldc(["diagnose", "--data-root", str(data_root), "--run", str(runs / "scratch" / share),
             "--init", str(init), "--device", "cpu"])
    (runs / "results").mkdir()
    ldc(["grad", "--data-root", str(data_root), "--init", str(init), "--device", "cpu",
         "--mc", "1", "--out", str(runs / "results" / "grad_iso.json")])
    ldc(["affine", "--data-root", str(data_root), "--init", str(init), "--device", "cpu",
         "--epochs", "2", "--seed", "42", "--out", str(runs / "results" / "affine_s42.json")])
    ldc(["mask-audit", "--data-root", str(data_root), "--out", str(runs / "results" / "mask_audit.json")])
    grad = json.loads((runs / "results" / "grad_iso.json").read_text())
    assert {"g_parallel_mean_normal", "g_parallel_mean_fault"} <= set(grad["sigreg_gradient"])
    ldc(["report", "--runs", str(runs), "--out", str(tmp_path / "results")])
    tables = (tmp_path / "results" / "tables.md").read_text()
    assert "Shared core" in tables and "Invertible affine control" in tables
    assert (tmp_path / "results" / "fig1_rank_vs_auc.png").exists()
