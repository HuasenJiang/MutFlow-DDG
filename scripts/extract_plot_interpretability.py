#!/usr/bin/env python3
"""Extract real MutFlow-DDG interpretability quantities and draw Figure 2.

The script targets the legacy dual-branch model.  It exports sample-level
attention, ablation, transport, uncertainty, and trajectory values before
drawing a publication-style 2x2 mechanistic interpretation figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import torch

from mutflow_ddg.config import load_config
from mutflow_ddg.device import describe_device, resolve_device
from mutflow_ddg.engine import load_torch_checkpoint
from mutflow_ddg.models import MutFlowDDG
from scripts.train import build_external_test_loaders


CLASS_ORDER = ("Stabilizing", "Neutral", "Destabilizing")
CLASS_COLORS = {
    "Stabilizing": "#6AAA45",
    "Neutral": "#13A6B8",
    "Destabilizing": "#F28E2B",
}
NAVY = "#17375E"
SLATE = "#536273"
RED = "#D84A4A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot MutFlow-DDG interpretability results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/checkpoints/mutflow_ddg/best.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="ssym",
        help="Configured external test set used for interpretation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures/interpretability"),
    )
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def mutation_class(target: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(
        target < -threshold,
        "Stabilizing",
        np.where(target > threshold, "Destabilizing", "Neutral"),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def downstream_from_branches(
    model: MutFlowDDG,
    sequence_wild: torch.Tensor,
    sequence_mutant: torch.Tensor,
    structure_wild: torch.Tensor,
    structure_mutant: torch.Tensor,
    mutation_features: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.state_fusion is None:
        raise ValueError("Modality ablation requires branch_mode=both")
    z0, _ = model.state_fusion(sequence_wild, structure_wild)
    z1, _ = model.state_fusion(sequence_mutant, structure_mutant)
    condition = model.condition_encoder(
        z_wild=z0,
        branch_wild=sequence_wild,
        branch_mutant=sequence_mutant,
        mutation_features=mutation_features,
    )
    z_flow, trajectory = model.flow.integrate(z0, condition)
    prediction = model.heads(z0, z1, z_flow, condition)
    return prediction, z0, z1, trajectory


def collect_interpretability(
    model: MutFlowDDG,
    loader,
    device: torch.device,
    dataset_name: str,
    neutral_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()
    sample_parts: list[pd.DataFrame] = []
    trajectory_parts: list[pd.DataFrame] = []
    latent_parts: dict[str, list[np.ndarray]] = {
        "z0": [],
        "z1": [],
        "z_flow": [],
        "trajectory": [],
    }
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            encoded = model.encode_pair(batch)
            required = {
                "sequence_wild",
                "sequence_mutant",
                "structure_wild",
                "structure_mutant",
                "attention_wild",
                "attention_mutant",
            }
            missing = required.difference(encoded)
            if missing:
                raise ValueError(
                    "The selected checkpoint does not expose dual-branch "
                    f"interpretability tensors: {sorted(missing)}"
                )

            full, z0, z1, trajectory = downstream_from_branches(
                model,
                encoded["sequence_wild"],
                encoded["sequence_mutant"],
                encoded["structure_wild"],
                encoded["structure_mutant"],
                batch.mutation_features,
            )
            z_flow = trajectory[:, -1]

            sequence_off, _, _, _ = downstream_from_branches(
                model,
                torch.zeros_like(encoded["sequence_wild"]),
                torch.zeros_like(encoded["sequence_mutant"]),
                encoded["structure_wild"],
                encoded["structure_mutant"],
                batch.mutation_features,
            )
            structure_off, _, _, _ = downstream_from_branches(
                model,
                encoded["sequence_wild"],
                encoded["sequence_mutant"],
                torch.zeros_like(encoded["structure_wild"]),
                torch.zeros_like(encoded["structure_mutant"]),
                batch.mutation_features,
            )

            target = batch.ddg.detach().cpu().numpy().astype(float)
            prediction = full["ddg"].detach().cpu().numpy().astype(float)
            log_variance = (
                full["log_variance"].detach().cpu().numpy().astype(float)
            )
            prediction_sequence_off = (
                sequence_off["ddg"].detach().cpu().numpy().astype(float)
            )
            prediction_structure_off = (
                structure_off["ddg"].detach().cpu().numpy().astype(float)
            )
            attention_wild = (
                encoded["attention_wild"].detach().cpu().numpy().astype(float)
            )
            attention_mutant = (
                encoded["attention_mutant"].detach().cpu().numpy().astype(float)
            )
            z0_np = z0.detach().cpu().numpy().astype(np.float32)
            z1_np = z1.detach().cpu().numpy().astype(np.float32)
            z_flow_np = z_flow.detach().cpu().numpy().astype(np.float32)
            trajectory_np = (
                trajectory.detach().cpu().numpy().astype(np.float32)
            )

            target_displacement = z1_np - z0_np
            flow_displacement = z_flow_np - z0_np
            target_norm = np.linalg.norm(target_displacement, axis=1)
            flow_norm = np.linalg.norm(flow_displacement, axis=1)
            direction_alignment = np.sum(
                target_displacement * flow_displacement, axis=1
            ) / np.maximum(target_norm * flow_norm, 1e-12)
            endpoint_error = np.linalg.norm(z_flow_np - z1_np, axis=1) / np.maximum(
                target_norm, 1e-12
            )
            segments = np.diff(trajectory_np, axis=1)
            segment_lengths = np.linalg.norm(segments, axis=2)
            path_length = segment_lengths.sum(axis=1)
            path_efficiency = flow_norm / np.maximum(path_length, 1e-12)

            distance = np.linalg.norm(
                trajectory_np - z1_np[:, None, :], axis=2
            )
            normalized_distance = distance / np.maximum(
                distance[:, [0]], 1e-12
            )
            progress = 1.0 - normalized_distance
            num_steps = trajectory_np.shape[1] - 1
            velocity = np.full_like(progress, np.nan)
            velocity[:, 1:] = segment_lengths * num_steps

            wild_reliance = attention_wild[:, 1] - attention_wild[:, 0]
            mutant_reliance = attention_mutant[:, 1] - attention_mutant[:, 0]
            delta_reliance = mutant_reliance - wild_reliance
            mean_reliance = 0.5 * (wild_reliance + mutant_reliance)
            absolute_error = np.abs(prediction - target)
            sequence_ablation_effect = (
                np.abs(prediction_sequence_off - target) - absolute_error
            )
            structure_ablation_effect = (
                np.abs(prediction_structure_off - target) - absolute_error
            )
            ablation_preference = (
                structure_ablation_effect - sequence_ablation_effect
            )
            classes = mutation_class(target, neutral_threshold)

            mutation_ids = batch.mutation_ids or [
                f"{dataset_name}:{batch_index}:{index}"
                for index in range(len(target))
            ]
            sample_part = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "mutation_id": mutation_ids,
                    "target_ddg": target,
                    "predicted_ddg": prediction,
                    "absolute_error": absolute_error,
                    "mutation_class": classes,
                    "log_variance": log_variance,
                    "predicted_sigma": np.exp(0.5 * log_variance),
                    "attention_wt_sequence": attention_wild[:, 0],
                    "attention_wt_structure": attention_wild[:, 1],
                    "attention_mut_sequence": attention_mutant[:, 0],
                    "attention_mut_structure": attention_mutant[:, 1],
                    "modality_reliance_wt": wild_reliance,
                    "modality_reliance_mut": mutant_reliance,
                    "modality_reliance_mean": mean_reliance,
                    "modality_reliance_delta": delta_reliance,
                    "prediction_sequence_off": prediction_sequence_off,
                    "prediction_structure_off": prediction_structure_off,
                    "sequence_ablation_delta_error": sequence_ablation_effect,
                    "structure_ablation_delta_error": structure_ablation_effect,
                    "ablation_preference_structure": ablation_preference,
                    "directional_alignment": direction_alignment,
                    "normalized_endpoint_error": endpoint_error,
                    "latent_path_length": path_length,
                    "path_efficiency": path_efficiency,
                }
            )
            sample_parts.append(sample_part)

            for step in range(trajectory_np.shape[1]):
                trajectory_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "mutation_id": mutation_ids,
                            "mutation_class": classes,
                            "step": step,
                            "time": step / num_steps,
                            "distance_to_z1": distance[:, step],
                            "normalized_distance_to_z1": normalized_distance[
                                :, step
                            ],
                            "normalized_progress": progress[:, step],
                            "velocity_norm": velocity[:, step],
                        }
                    )
                )

            latent_parts["z0"].append(z0_np)
            latent_parts["z1"].append(z1_np)
            latent_parts["z_flow"].append(z_flow_np)
            latent_parts["trajectory"].append(trajectory_np)

            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"Extracted {dataset_name}: batch {batch_index}/{total_batches}",
                    flush=True,
                )

    samples = pd.concat(sample_parts, ignore_index=True)
    trajectories = pd.concat(trajectory_parts, ignore_index=True)
    latents = {
        key: np.concatenate(parts, axis=0) for key, parts in latent_parts.items()
    }
    latents["mutation_id"] = samples["mutation_id"].astype(str).to_numpy()
    return samples, trajectories, latents


def coverage_curve(
    target: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
) -> pd.DataFrame:
    absolute_error = np.abs(target - prediction)
    normal = NormalDist()
    nominal = np.linspace(0.10, 0.95, 18)
    rows = []
    for level in nominal:
        quantile = normal.inv_cdf((1.0 + float(level)) / 2.0)
        observed = float(np.mean(absolute_error <= quantile * sigma))
        standard_error = math.sqrt(
            max(observed * (1.0 - observed), 1e-12) / len(target)
        )
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": observed,
                "ci95_low": max(0.0, observed - 1.96 * standard_error),
                "ci95_high": min(1.0, observed + 1.96 * standard_error),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage_curve(
    absolute_error: np.ndarray,
    uncertainty: np.ndarray,
) -> pd.DataFrame:
    n = len(absolute_error)
    fractions = np.linspace(0.05, 1.0, 40)
    confidence_order = np.argsort(uncertainty)
    oracle_order = np.argsort(absolute_error)
    rows = []
    for fraction in fractions:
        count = max(1, int(math.ceil(n * fraction)))
        rows.append(
            {
                "coverage": fraction,
                "uncertainty_ranked_mae": float(
                    np.mean(absolute_error[confidence_order[:count]])
                ),
                "oracle_mae": float(
                    np.mean(absolute_error[oracle_order[:count]])
                ),
                "random_expected_mae": float(np.mean(absolute_error)),
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.grid(True, color="#D9E0E8", lw=0.7, alpha=0.62, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#AEB9C6")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=11.5, colors="#405060")


def panel_label(ax: plt.Axes, letter: str, x: float = -0.14) -> None:
    ax.text(
        x,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
        clip_on=False,
    )


def draw_figure(
    samples: pd.DataFrame,
    trajectories: pd.DataFrame,
    coverage: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    seed: int,
    footer_text: str = (
        "All panels use measured held-out predictions and internal states; "
        "shaded regions in B denote the interquartile range."
    ),
    stem_name: str = "figure2_interpretability",
    schematic_label: str | None = None,
    extra_attention_panels: bool = False,
) -> dict[str, float]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rng = np.random.default_rng(seed)
    figure_height = 15.2 if extra_attention_panels else 10.8
    figure_rows = 3 if extra_attention_panels else 2
    fig = plt.figure(figsize=(15.8, figure_height), dpi=dpi)
    grid = fig.add_gridspec(
        figure_rows,
        2,
        left=0.065,
        right=0.965,
        top=0.925,
        bottom=0.095,
        wspace=0.24,
        hspace=0.31,
    )
    a_grid = grid[0, 0].subgridspec(1, 2, width_ratios=(0.92, 1.08), wspace=0.38)
    ax_a1 = fig.add_subplot(a_grid[0, 0])
    ax_a2 = fig.add_subplot(a_grid[0, 1])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    if extra_attention_panels:
        ax_e = fig.add_subplot(grid[2, 0])
        ax_f = fig.add_subplot(grid[2, 1])
        axes = (ax_a1, ax_a2, ax_b, ax_c, ax_d, ax_e, ax_f)
    else:
        ax_e = None
        ax_f = None
        axes = (ax_a1, ax_a2, ax_b, ax_c, ax_d)
    for axis in axes:
        style_axis(axis)

    # A1: mutation-induced change in sequence/structure reliance.
    violin_values = [
        samples.loc[
            samples["mutation_class"] == label, "modality_reliance_delta"
        ].to_numpy()
        for label in CLASS_ORDER
    ]
    violin = ax_a1.violinplot(
        violin_values,
        positions=np.arange(1, 4),
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, label in zip(violin["bodies"], CLASS_ORDER):
        body.set_facecolor(CLASS_COLORS[label])
        body.set_edgecolor("white")
        body.set_alpha(0.55)
    for position, label, values in zip(
        np.arange(1, 4), CLASS_ORDER, violin_values
    ):
        jitter = rng.normal(0.0, 0.055, len(values))
        ax_a1.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=8,
            color=CLASS_COLORS[label],
            alpha=0.28,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_a1.plot([position, position], [q1, q3], color=NAVY, lw=3.0, zorder=4)
        ax_a1.scatter(
            [position], [median], s=34, color="white", edgecolor=NAVY, zorder=5
        )
    ax_a1.axhline(0.0, color=SLATE, ls=(0, (5, 4)), lw=1.2)
    ax_a1.set_xticks([1, 2, 3], ["Stab.", "Neutral", "Destab."])
    ax_a1.set_ylabel(
        r"Mutation-induced modality shift  $\Delta r$", fontsize=12.8, color=NAVY
    )
    ax_a1.set_title("Modality shift", fontsize=16.5, fontweight="bold", color=NAVY)
    panel_label(ax_a1, "A", x=-0.22)

    # A2: agreement between attention and causal modality ablation.
    for label in CLASS_ORDER:
        subset = samples[samples["mutation_class"] == label]
        ax_a2.scatter(
            subset["modality_reliance_mean"],
            subset["ablation_preference_structure"],
            s=16,
            color=CLASS_COLORS[label],
            alpha=0.45,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            label=label,
            zorder=3,
        )
    x = samples["modality_reliance_mean"].to_numpy()
    y = samples["ablation_preference_structure"].to_numpy()
    if np.std(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
        fit_x = np.linspace(np.min(x), np.max(x), 120)
        ax_a2.plot(fit_x, slope * fit_x + intercept, color=RED, lw=1.8, zorder=4)
    rho_attention = spearman(x, y)
    ax_a2.axhline(0.0, color=SLATE, ls=(0, (5, 4)), lw=1.0)
    ax_a2.axvline(0.0, color=SLATE, ls=(0, (5, 4)), lw=1.0)
    ax_a2.set_xlabel("Mean structure reliance", fontsize=12.8, color=NAVY)
    ax_a2.set_ylabel("Structure−sequence\nablation effect", fontsize=12.8, color=NAVY)
    ax_a2.set_title(
        "Attention–ablation agreement", fontsize=16.5, fontweight="bold", color=NAVY
    )
    ax_a2.text(
        0.04,
        0.95,
        rf"Spearman $\rho$ = {rho_attention:.3f}",
        transform=ax_a2.transAxes,
        va="top",
        fontsize=11.5,
        color="#253442",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": (1, 1, 1, 0.88),
            "edgecolor": "#B8C5D1",
        },
    )

    # B: normalized progression toward the encoded mutant state.
    for label in CLASS_ORDER:
        subset = trajectories[trajectories["mutation_class"] == label]
        pivot = subset.pivot(
            index="mutation_id", columns="step", values="normalized_progress"
        )
        times = np.array(sorted(subset["time"].unique()))
        values = pivot.to_numpy()
        median = np.nanmedian(values, axis=0)
        lower = np.nanquantile(values, 0.25, axis=0)
        upper = np.nanquantile(values, 0.75, axis=0)
        ax_b.fill_between(times, lower, upper, color=CLASS_COLORS[label], alpha=0.18)
        ax_b.plot(
            times,
            median,
            marker="o",
            ms=5.0,
            lw=2.2,
            color=CLASS_COLORS[label],
            label=label,
            zorder=4,
        )
    ideal_time = np.linspace(0.0, 1.0, 100)
    ax_b.plot(
        ideal_time,
        ideal_time,
        color=SLATE,
        lw=1.4,
        ls=(0, (5, 4)),
        label="Linear reference",
    )
    ax_b.set_xlabel("Normalized Euler time", fontsize=13.5, color=NAVY)
    ax_b.set_ylabel(r"Progress toward $z_1$", fontsize=13.5, color=NAVY)
    ax_b.set_title(
        "Normalized flow progression", fontsize=18, fontweight="bold", color=NAVY
    )
    ax_b.legend(frameon=False, fontsize=10.5, loc="lower right")
    panel_label(ax_b, "B")

    # C: direct fidelity of the integrated displacement.
    alignment = samples["directional_alignment"].to_numpy()
    endpoint = samples["normalized_endpoint_error"].to_numpy()
    absolute_error = samples["absolute_error"].to_numpy()
    error_max = float(np.quantile(absolute_error, 0.98))
    scatter = ax_c.scatter(
        alignment,
        endpoint,
        c=absolute_error,
        cmap="magma_r",
        vmin=0.0,
        vmax=error_max,
        s=20,
        alpha=0.64,
        linewidths=0.20,
        edgecolors=(1, 1, 1, 0.55),
        rasterized=True,
        zorder=3,
    )
    ax_c.axvline(1.0, color=SLATE, lw=1.1, ls=(0, (5, 4)))
    ax_c.axhline(0.0, color=SLATE, lw=1.1, ls=(0, (5, 4)))
    ax_c.set_xlim(max(-1.0, float(np.quantile(alignment, 0.005)) - 0.03), 1.02)
    endpoint_upper = max(0.1, float(np.quantile(endpoint, 0.99)) * 1.08)
    ax_c.set_ylim(-0.02 * endpoint_upper, endpoint_upper)
    ax_c.set_xlabel(
        r"Directional alignment  $\cos(\Delta z_{flow},\Delta z_{target})$",
        fontsize=13.2,
        color=NAVY,
    )
    ax_c.set_ylabel("Normalized endpoint error", fontsize=13.5, color=NAVY)
    ax_c.set_title(
        "Latent transport fidelity", fontsize=18, fontweight="bold", color=NAVY
    )
    rho_transport = spearman(endpoint, absolute_error)
    ax_c.text(
        0.04,
        0.95,
        "Median alignment = "
        f"{np.median(alignment):.3f}\n"
        rf"$\rho$(endpoint error, $|e|$) = {rho_transport:.3f}",
        transform=ax_c.transAxes,
        va="top",
        fontsize=11.5,
        color="#253442",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": (1, 1, 1, 0.90),
            "edgecolor": "#B8C5D1",
        },
    )
    colorbar = fig.colorbar(scatter, ax=ax_c, fraction=0.047, pad=0.025, extend="max")
    colorbar.set_label(r"Absolute DDG error (kcal mol$^{-1}$)", fontsize=11.5)
    colorbar.ax.tick_params(labelsize=10)
    panel_label(ax_c, "C")

    # D: interval coverage plus risk-coverage inset.
    empirical = coverage["empirical_coverage"].to_numpy()
    nominal = coverage["nominal_coverage"].to_numpy()
    lower_error = empirical - coverage["ci95_low"].to_numpy()
    upper_error = coverage["ci95_high"].to_numpy() - empirical
    ax_d.plot([0, 1], [0, 1], color=SLATE, lw=1.4, ls=(0, (5, 4)), label="Ideal")
    ax_d.errorbar(
        nominal,
        empirical,
        yerr=np.vstack([lower_error, upper_error]),
        color="#168AAD",
        marker="o",
        ms=5.5,
        lw=2.0,
        capsize=2.5,
        label="MutFlow-DDG",
        zorder=4,
    )
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlabel("Nominal prediction-interval coverage", fontsize=13.0, color=NAVY)
    ax_d.set_ylabel("Empirical coverage", fontsize=13.5, color=NAVY)
    ax_d.set_title(
        "Uncertainty calibration", fontsize=18, fontweight="bold", color=NAVY
    )
    ax_d.legend(frameon=False, fontsize=11, loc="upper left")
    calibration_error = float(np.mean(np.abs(empirical - nominal)))
    rho_uncertainty = spearman(
        samples["predicted_sigma"].to_numpy(), absolute_error
    )
    ax_d.text(
        0.96,
        0.95,
        f"Mean coverage gap = {calibration_error:.3f}\n"
        rf"$\rho(\sigma,|e|)$ = {rho_uncertainty:.3f}",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color="#253442",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": (1, 1, 1, 0.88),
            "edgecolor": "#B8C5D1",
        },
    )
    risk_axis = inset_axes(
        ax_d,
        width="47%",
        height="39%",
        loc="lower right",
        borderpad=1.45,
    )
    risk_axis.plot(
        risk_coverage["coverage"],
        risk_coverage["uncertainty_ranked_mae"],
        color="#168AAD",
        lw=1.8,
        label="Uncertainty",
    )
    risk_axis.plot(
        risk_coverage["coverage"],
        risk_coverage["oracle_mae"],
        color="#6AAA45",
        lw=1.5,
        ls=(0, (4, 3)),
        label="Oracle",
    )
    risk_axis.plot(
        risk_coverage["coverage"],
        risk_coverage["random_expected_mae"],
        color="#7A8795",
        lw=1.3,
        ls=":",
        label="Random",
    )
    risk_axis.set_title("Risk–coverage", fontsize=10.5, fontweight="bold")
    risk_axis.set_xlabel("Retained fraction", fontsize=9.2)
    risk_axis.set_ylabel("MAE", fontsize=9.2)
    risk_axis.tick_params(labelsize=8.5)
    risk_axis.grid(True, color="#D9E0E8", lw=0.5, alpha=0.55)
    for spine in risk_axis.spines.values():
        spine.set_color("#B8C5D1")
    risk_axis.legend(frameon=False, fontsize=7.3, loc="upper left")
    panel_label(ax_d, "D")

    attention_state_rho = float("nan")
    attention_shift_rho = float("nan")
    if extra_attention_panels:
        required_attention = {
            "attention_wt_structure",
            "attention_mut_structure",
            "modality_reliance_delta",
            "target_ddg",
        }
        missing_attention = required_attention.difference(samples.columns)
        if missing_attention:
            raise ValueError(
                "Extra attention panels require columns: "
                + ", ".join(sorted(missing_attention))
            )
        assert ax_e is not None and ax_f is not None

        # E: how structure attention changes from WT to mutant state.
        for label in CLASS_ORDER:
            subset = samples[samples["mutation_class"] == label]
            ax_e.scatter(
                subset["attention_wt_structure"],
                subset["attention_mut_structure"],
                s=23,
                color=CLASS_COLORS[label],
                alpha=0.46,
                linewidths=0.25,
                edgecolors="white",
                rasterized=True,
                label=label,
                zorder=3,
            )
            center_x = float(subset["attention_wt_structure"].mean())
            center_y = float(subset["attention_mut_structure"].mean())
            ax_e.scatter(
                [center_x],
                [center_y],
                marker="D",
                s=72,
                color=CLASS_COLORS[label],
                edgecolor=NAVY,
                linewidth=0.9,
                zorder=5,
            )
        ax_e.plot(
            [0, 1],
            [0, 1],
            color=SLATE,
            lw=1.4,
            ls=(0, (5, 4)),
            zorder=2,
        )
        ax_e.fill_between(
            [0, 1],
            [0, 1],
            [1, 1],
            color="#7E57C2",
            alpha=0.045,
            zorder=1,
        )
        ax_e.fill_between(
            [0, 1],
            [0, 0],
            [0, 1],
            color="#2A9D8F",
            alpha=0.045,
            zorder=1,
        )
        ax_e.text(
            0.04,
            0.94,
            "Structure attention increases",
            transform=ax_e.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            color="#7652A6",
        )
        ax_e.text(
            0.96,
            0.06,
            "Structure attention decreases",
            transform=ax_e.transAxes,
            ha="right",
            va="bottom",
            fontsize=11.2,
            color="#397E75",
        )
        attention_wt = samples["attention_wt_structure"].to_numpy()
        attention_mut = samples["attention_mut_structure"].to_numpy()
        attention_state_rho = spearman(attention_wt, attention_mut)
        ax_e.set_xlim(0.0, 1.0)
        ax_e.set_ylim(0.0, 1.0)
        ax_e.set_aspect("equal", adjustable="box")
        ax_e.set_xlabel(r"WT structure attention  $\alpha^{WT}_{str}$", fontsize=13.5, color=NAVY)
        ax_e.set_ylabel(r"MUT structure attention  $\alpha^{MUT}_{str}$", fontsize=13.5, color=NAVY)
        ax_e.set_title(
            "WT–MUT attention transition", fontsize=18, fontweight="bold", color=NAVY
        )
        ax_e.legend(frameon=False, fontsize=10.5, loc="upper right")
        panel_label(ax_e, "E")

        # F: continuous association between mutation effect and attention shift.
        target_ddg = samples["target_ddg"].to_numpy()
        structure_shift = 0.5 * samples["modality_reliance_delta"].to_numpy()
        for label in CLASS_ORDER:
            mask = samples["mutation_class"].to_numpy() == label
            ax_f.scatter(
                target_ddg[mask],
                structure_shift[mask],
                s=18,
                color=CLASS_COLORS[label],
                alpha=0.32,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
        quantile_edges = np.unique(np.quantile(target_ddg, np.linspace(0, 1, 11)))
        bin_centers: list[float] = []
        bin_means: list[float] = []
        bin_errors: list[float] = []
        for index, (left, right) in enumerate(
            zip(quantile_edges[:-1], quantile_edges[1:])
        ):
            if index == len(quantile_edges) - 2:
                mask = (target_ddg >= left) & (target_ddg <= right)
            else:
                mask = (target_ddg >= left) & (target_ddg < right)
            if np.count_nonzero(mask) < 2:
                continue
            values = structure_shift[mask]
            bin_centers.append(float(np.mean(target_ddg[mask])))
            bin_means.append(float(np.mean(values)))
            bin_errors.append(float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))))
        ax_f.errorbar(
            bin_centers,
            bin_means,
            yerr=bin_errors,
            color=NAVY,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=NAVY,
            markersize=6.2,
            lw=2.2,
            capsize=3.0,
            label="Binned mean ± 95% CI",
            zorder=5,
        )
        ax_f.axhline(0.0, color=SLATE, lw=1.1, ls=(0, (5, 4)))
        ax_f.axvline(0.0, color=SLATE, lw=1.1, ls=(0, (5, 4)))
        attention_shift_rho = spearman(target_ddg, structure_shift)
        ax_f.text(
            0.04,
            0.95,
            rf"Spearman $\rho$ = {attention_shift_rho:.3f}",
            transform=ax_f.transAxes,
            va="top",
            fontsize=11.8,
            color="#253442",
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": (1, 1, 1, 0.88),
                "edgecolor": "#B8C5D1",
            },
        )
        ax_f.set_xlabel(r"Experimental $\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=13.5, color=NAVY)
        ax_f.set_ylabel(
            r"Mutation-induced structure-attention shift  $\Delta\alpha_{str}$",
            fontsize=13.0,
            color=NAVY,
        )
        ax_f.set_title(
            "Attention–effect coupling", fontsize=18, fontweight="bold", color=NAVY
        )
        ax_f.legend(frameon=False, fontsize=10.5, loc="lower right")
        panel_label(ax_f, "F")

    fig.text(
        0.5,
        0.025,
        footer_text,
        ha="center",
        fontsize=11.5,
        fontstyle="italic",
        color="#657483",
    )
    if schematic_label:
        fig.text(
            0.965,
            0.972,
            schematic_label,
            ha="right",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            color="#9B4A55",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#FFF4F3",
                "edgecolor": "#DDA7A4",
                "linewidth": 0.9,
            },
        )

    stem = out_dir / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    plt.close(fig)

    aurc = float(
        np.trapz(
            risk_coverage["uncertainty_ranked_mae"],
            risk_coverage["coverage"],
        )
    )
    summary = {
        "samples": int(len(samples)),
        "attention_ablation_spearman": rho_attention,
        "delta_reliance_ddg_spearman": spearman(
            samples["modality_reliance_delta"].to_numpy(),
            samples["target_ddg"].to_numpy(),
        ),
        "median_directional_alignment": float(np.median(alignment)),
        "median_normalized_endpoint_error": float(np.median(endpoint)),
        "endpoint_error_absolute_error_spearman": rho_transport,
        "uncertainty_absolute_error_spearman": rho_uncertainty,
        "mean_absolute_coverage_gap": calibration_error,
        "risk_coverage_auc": aurc,
    }
    if extra_attention_panels:
        summary.update(
            {
                "wt_mut_structure_attention_spearman": attention_state_rho,
                "structure_attention_shift_ddg_spearman": attention_shift_rho,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    selected_tests = [
        item for item in config.data.external_tests if item.name == args.dataset
    ]
    if not selected_tests:
        available = [item.name for item in config.data.external_tests]
        raise ValueError(
            f"Unknown external test {args.dataset!r}; available: {available}"
        )
    config.data.external_tests = selected_tests
    device = resolve_device(args.device)
    print(f"Using device: {describe_device(device)}", flush=True)

    checkpoint = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    model = MutFlowDDG(config.model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    loaders = build_external_test_loaders(config)
    samples, trajectories, latents = collect_interpretability(
        model,
        loaders[args.dataset],
        device,
        args.dataset,
        config.loss.neutral_threshold,
    )

    coverage = coverage_curve(
        samples["target_ddg"].to_numpy(),
        samples["predicted_ddg"].to_numpy(),
        samples["predicted_sigma"].to_numpy(),
    )
    risk_coverage = risk_coverage_curve(
        samples["absolute_error"].to_numpy(),
        samples["predicted_sigma"].to_numpy(),
    )

    samples.to_csv(args.out_dir / "interpretability_samples.csv", index=False)
    trajectories.to_csv(
        args.out_dir / "interpretability_trajectory.csv", index=False
    )
    coverage.to_csv(args.out_dir / "uncertainty_coverage.csv", index=False)
    risk_coverage.to_csv(args.out_dir / "risk_coverage.csv", index=False)
    np.savez_compressed(args.out_dir / "interpretability_latents.npz", **latents)

    summary = draw_figure(
        samples,
        trajectories,
        coverage,
        risk_coverage,
        args.out_dir,
        args.dpi,
        args.seed,
    )
    summary.update(
        {
            "dataset": args.dataset,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "neutral_threshold": float(config.loss.neutral_threshold),
        }
    )
    with (args.out_dir / "interpretability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved interpretability outputs to: {args.out_dir.resolve()}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
