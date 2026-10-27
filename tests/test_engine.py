from pathlib import Path

import torch
from torch import nn

from mutflow_ddg.config import ExperimentConfig
from mutflow_ddg.data.synthetic import make_synthetic_batch
from mutflow_ddg.engine import Trainer, progress_bars_enabled


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return {"ddg": self.weight.expand_as(batch.ddg)}


class RecordingObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.reverse_outputs = []

    def forward(self, output, batch, reverse_output=None):
        self.reverse_outputs.append(reverse_output)
        loss = (output["ddg"] - batch.ddg).abs().mean()
        return loss, {"regression": loss, "total": loss}


def test_progress_bars_can_be_disabled_by_config_or_environment(monkeypatch):
    config = ExperimentConfig()
    config.training.show_progress = False
    assert progress_bars_enabled(config) is False

    monkeypatch.setenv("MUTFLOW_PROGRESS", "1")
    assert progress_bars_enabled(config) is True

    monkeypatch.setenv("MUTFLOW_PROGRESS", "0")
    config.training.show_progress = True
    assert progress_bars_enabled(config) is False


def test_validation_does_not_run_separate_reverse_pass():
    config = ExperimentConfig()
    config.loss.symmetry = 0.1
    model = CountingModel()
    trainer = Trainer(model, config, torch.device("cpu"))
    objective = RecordingObjective()
    trainer.objective = objective
    batch = make_synthetic_batch()

    trainer._step(batch, training=False)
    assert model.calls == 1
    assert objective.reverse_outputs[-1] is None

    model.calls = 0
    trainer._step(batch, training=True)
    assert model.calls == 2
    assert objective.reverse_outputs[-1] is not None


def test_disabled_symmetry_avoids_reverse_training_pass():
    config = ExperimentConfig()
    config.loss.symmetry = 0.0
    model = CountingModel()
    trainer = Trainer(model, config, torch.device("cpu"))
    objective = RecordingObjective()
    trainer.objective = objective

    trainer._step(make_synthetic_batch(), training=True)

    assert model.calls == 1
    assert objective.reverse_outputs[-1] is None


def test_fit_selects_checkpoint_by_validation_l1(tmp_path: Path):
    config = ExperimentConfig()
    config.training.epochs = 2
    config.training.top_k_checkpoints = 2
    config.training.show_progress = False
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    results = iter(
        [
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 0.1, "l1": 2.0, "mae": 0.5, "mse": 0.25, "pcc": 0.0},
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 10.0, "l1": 1.0, "mae": 2.0, "mse": 4.0, "pcc": 0.0},
            {"total": 10.0, "l1": 1.0, "mae": 2.0, "mse": 4.0, "pcc": 0.0},
        ]
    )
    trainer.run_epoch = lambda *args, **kwargs: next(results)
    validation_loader = [make_synthetic_batch()]

    trainer.fit([], validation_loader)

    try:
        checkpoint = torch.load(
            tmp_path / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert checkpoint["loss"] == 1.0
    assert checkpoint["monitor"] == "l1"
    manifest = (tmp_path / "top5_checkpoints.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "top_epoch_0002_val_1.00000000.pt" in manifest[1]
    assert "top_epoch_0001_val_2.00000000.pt" in manifest[2]


def test_fit_resumes_after_last_completed_epoch(tmp_path: Path):
    metrics = {
        "total": 1.0,
        "l1": 1.0,
        "mae": 1.0,
        "mse": 1.0,
        "pcc": 0.0,
    }
    config = ExperimentConfig()
    config.training.epochs = 1
    config.training.top_k_checkpoints = 2
    config.training.show_progress = False
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    trainer.run_epoch = lambda *args, **kwargs: metrics
    validation_loader = [make_synthetic_batch()]
    trainer.fit([], validation_loader)

    resumed_config = ExperimentConfig()
    resumed_config.training.epochs = 2
    resumed_config.training.top_k_checkpoints = 2
    resumed_config.training.show_progress = False
    resumed_config.training.checkpoint_dir = str(tmp_path)
    resumed_trainer = Trainer(
        CountingModel(),
        resumed_config,
        torch.device("cpu"),
    )
    epochs = []

    def run_epoch(loader, training, epoch):
        epochs.append((training, epoch))
        return metrics

    resumed_trainer.run_epoch = run_epoch
    resumed_trainer.fit(
        [],
        validation_loader,
        resume_path=tmp_path / "last.pt",
    )

    assert epochs[:2] == [(True, 2), (False, 2)]
    try:
        checkpoint = torch.load(
            tmp_path / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert "scheduler_state_dict" in checkpoint
    assert len(checkpoint["top_checkpoints"]) == 2


def test_fit_predicts_external_dataset_with_each_top_model(tmp_path: Path):
    config = ExperimentConfig()
    config.training.epochs = 2
    config.training.top_k_checkpoints = 2
    config.training.show_progress = False
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    results = iter(
        [
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
            {"total": 1.0, "l1": 1.0, "mae": 1.0, "mse": 1.0, "pcc": 0.0},
        ]
    )
    trainer.run_epoch = lambda *args, **kwargs: next(results)
    validation_loader = [make_synthetic_batch(batch_size=3)]
    external_loader = [make_synthetic_batch(batch_size=2)]
    trainer.fit(
        [],
        validation_loader,
        external_test_loaders={"ptmul-NR": external_loader},
    )

    metric_lines = (tmp_path / "top5_metrics.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(metric_lines) == 5
    assert sum(",validation," in line for line in metric_lines) == 2
    assert sum(",ptmul-NR," in line for line in metric_lines) == 2

    prediction_lines = (tmp_path / "top5_predictions.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(prediction_lines) == 1 + 2 * (3 + 2)

    try:
        checkpoint = torch.load(
            tmp_path / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert checkpoint["top_rank"] == 1
    assert checkpoint["external_tests"]["ptmul-NR"]["samples"] == 2


def test_evaluate_saved_checkpoints_does_not_retrain(tmp_path: Path):
    config = ExperimentConfig()
    config.training.epochs = 1
    config.training.top_k_checkpoints = 1
    config.training.show_progress = False
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    metrics = {
        "total": 1.0,
        "l1": 1.0,
        "mae": 1.0,
        "mse": 1.0,
        "pcc": 0.0,
    }
    trainer.run_epoch = lambda *args, **kwargs: metrics
    validation_loader = [make_synthetic_batch(batch_size=2)]
    trainer.fit([], validation_loader)

    evaluation_trainer = Trainer(
        CountingModel(),
        config,
        torch.device("cpu"),
    )
    evaluation_trainer.run_epoch = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("evaluate-only must not call the training epoch loop")
    )
    result = evaluation_trainer.evaluate_saved_checkpoints(
        validation_loader,
        {"external": [make_synthetic_batch(batch_size=1)]},
    )

    assert result["samples"] == 2
    metric_lines = (tmp_path / "top5_metrics.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert any(",external," in line for line in metric_lines)


def test_fit_runs_periodic_and_final_best_model_external_tests(
    tmp_path: Path,
):
    config = ExperimentConfig()
    config.training.epochs = 10
    config.training.show_progress = False
    config.training.external_test_interval = 10
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    metrics = {
        "total": 1.0,
        "l1": 1.0,
        "mae": 1.0,
        "mse": 1.0,
        "pcc": 0.0,
    }
    trainer.run_epoch = lambda *args, **kwargs: metrics
    external_calls = []

    def evaluate_external(loader, epoch, dataset_name):
        external_calls.append((loader, epoch, dataset_name))
        return {
            "samples": 1,
            "mae": float(epoch),
            "mse": float(epoch),
            "rmse": float(epoch),
            "pcc": 0.5,
            "spearman": 0.4,
        }

    trainer.evaluate_regression = evaluate_external
    validation_loader = [make_synthetic_batch(batch_size=2)]
    external_loader = [make_synthetic_batch(batch_size=2)]
    trainer.fit(
        [],
        validation_loader,
        external_test_loaders={"ptmul-NR": external_loader},
    )

    assert external_calls == [(external_loader, 10, "ptmul-NR")]
    lines = (tmp_path / "top5_metrics.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1 + 5 * 2


def test_fit_retains_only_the_five_lowest_validation_losses(tmp_path: Path):
    config = ExperimentConfig()
    config.training.epochs = 6
    config.training.top_k_checkpoints = 5
    config.training.show_progress = False
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    validation_losses = iter([0.6, 0.1, 0.5, 0.2, 0.4, 0.3])
    current_validation_loss = None

    def run_epoch(loader, training, epoch):
        nonlocal current_validation_loss
        if training:
            current_validation_loss = next(validation_losses)
            loss = 1.0
        else:
            loss = current_validation_loss
        return {
            "total": loss,
            "l1": loss,
            "mae": loss,
            "mse": loss**2,
            "pcc": 0.0,
        }

    trainer.run_epoch = run_epoch
    trainer.fit([], [make_synthetic_batch()])

    manifest = (tmp_path / "top5_checkpoints.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(manifest) == 6
    assert [float(line.rsplit(",", 1)[1]) for line in manifest[1:]] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert len(list(tmp_path.glob("top_epoch_*_val_*.pt"))) == 5


def test_resume_archives_a_legacy_single_best_checkpoint(tmp_path: Path):
    config = ExperimentConfig()
    config.training.checkpoint_dir = str(tmp_path)
    trainer = Trainer(CountingModel(), config, torch.device("cpu"))
    torch.save(
        {
            "epoch": 7,
            "loss": 0.4,
            "metrics": {"l1": 0.4},
        },
        tmp_path / "best.pt",
    )

    entries = trainer._restore_top_checkpoints(
        tmp_path,
        {"top_checkpoints": []},
    )

    assert entries == [
        {
            "epoch": 7,
            "loss": 0.4,
            "path": "top_epoch_0007_val_0.40000000.pt",
        }
    ]
    assert (tmp_path / entries[0]["path"]).is_file()


def test_directional_evaluation_reports_total_direct_and_reverse():
    config = ExperimentConfig()
    model = CountingModel()
    trainer = Trainer(model, config, torch.device("cpu"))
    batch = make_synthetic_batch(batch_size=4)

    metrics = trainer.evaluate_directional_regression(
        [batch],
        epoch=1,
        dataset_name="s669",
    )

    assert list(metrics) == ["total", "direct", "reverse"]
    assert metrics["total"]["samples"] == 8
    assert metrics["direct"]["samples"] == 4
    assert metrics["reverse"]["samples"] == 4
    assert model.calls == 2
