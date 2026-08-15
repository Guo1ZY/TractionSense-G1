#!/usr/bin/env python3
"""Analyze four matched real G1 Hall trials without converting Hall to force.

This is an exploratory single-trial-per-condition analysis.  Time windows are
descriptive samples from one trajectory, not independent biological/robotic
replicates, and are never randomly split to claim classifier generalization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend, welch


ROOT = Path("/home/mosense/guo_1/vola_sensor/real_data")
OUT = Path("/home/mosense/guo_1/vola_sensor/analysis/four_condition_20260812")
TRIALS = {
    "high_walkrun": ROOT
    / "high_walkrun_v040_20260812_1740_r01"
    / "high_walkrun_v040_20260812_1740_r01_high.npz",
    "low_walkrun": ROOT
    / "low_walkrun_v040_20260812_1806_r02"
    / "low_walkrun_v040_20260812_1806_r02_low.npz",
    "high_waist": ROOT
    / "high_waistwalk_v040_20260812_1742_r01"
    / "high_waistwalk_v040_20260812_1742_r01_high.npz",
    "low_waist": ROOT
    / "low_waistwalk_v040_20260812_1803_r02"
    / "low_waistwalk_v040_20260812_1803_r02_low.npz",
}
ORDER = ("high_walkrun", "low_walkrun", "high_waist", "low_waist")
DISPLAY = {
    "high_walkrun": "High μ · walk–run",
    "low_walkrun": "Low μ · walk–run",
    "high_waist": "High μ · waist-walk",
    "low_waist": "Low μ · waist-walk",
}
COLORS = {
    "high_walkrun": "#3B6FB6",
    "low_walkrun": "#D8892B",
    "high_waist": "#76A5D8",
    "low_waist": "#E7AD62",
}
LINESTYLES = {
    "high_walkrun": "-",
    "low_walkrun": "-",
    "high_waist": "--",
    "low_waist": "--",
}


def channel_valid(hall: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    if hall.ndim != 4 or hall.shape[1:] != (2, 15, 3):
        raise ValueError(f"expected Hall [N,2,15,3], got {hall.shape}")
    if temperature.shape != hall.shape[:-1]:
        raise ValueError(f"temperature shape mismatch: {temperature.shape}")
    return (temperature > -9000) & np.any(hall != 0, axis=-1)


def longest_true_slice(mask: np.ndarray) -> slice:
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    if not len(starts):
        raise ValueError("no valid samples")
    index = int(np.argmax(stops - starts))
    return slice(int(starts[index]), int(stops[index]))


def load_trial(name: str, path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        hall = data["hall_xyz"].astype(np.float64)
        temperature = data["temperature_x10"].astype(np.int32)
        transport_valid = data["valid"].astype(bool)
        timestamp_ns = data["publish_monotonic_ns"].astype(np.int64)
        metadata = json.loads(str(data["metadata_json"]))
    valid_site = channel_valid(hall, temperature)
    valid_row = np.all(valid_site, axis=(1, 2)) & np.all(transport_valid, axis=1)
    dt = np.diff(timestamp_ns[valid_row]).astype(np.float64) / 1.0e9
    fs = float(1.0 / np.median(dt))
    if not 45.0 <= fs <= 52.0:
        raise ValueError(f"{name}: unexpected paired rate {fs}")
    # A channel reconnect can leave a transient immediately before/after the
    # explicit zero/sentinel rows. Keep raw validity intact for QA, but use a
    # transparent 1 s guard band for signal statistics.
    guard = int(round(fs))
    bad = (~valid_row).astype(np.int16)
    near_bad = np.convolve(
        bad, np.ones(2 * guard + 1, dtype=np.int16), mode="same"
    ) > 0
    analysis_valid_row = valid_row & (~near_bad)
    return {
        "name": name,
        "path": path,
        "hall": hall,
        "temperature": temperature,
        "timestamp_ns": timestamp_ns,
        "valid_site": valid_site,
        "valid_row": valid_row,
        "analysis_valid_row": analysis_valid_row,
        "metadata": metadata,
        "fs": fs,
    }


def make_windows(trial: dict, window_s: float = 1.0, step_s: float = 0.25) -> list[dict]:
    hall = trial["hall"]
    valid = trial["analysis_valid_row"]
    fs = trial["fs"]
    width = int(round(window_s * fs))
    step = int(round(step_s * fs))
    edge = int(round(fs))
    windows = []
    for start in range(edge, len(hall) - edge - width + 1, step):
        stop = start + width
        if not bool(np.all(valid[start:stop])):
            continue
        values = hall[start:stop]
        centered = detrend(values, axis=0, type="linear")
        delta = np.diff(values, axis=0)
        site_rms = np.sqrt(np.mean(centered**2, axis=(0, 3)))
        site_delta_rms = np.sqrt(np.mean(delta**2, axis=(0, 3)))
        windows.append(
            {
                "start": start,
                "time_s": (trial["timestamp_ns"][start] - trial["timestamp_ns"][0])
                / 1.0e9,
                "dynamic_rms": float(np.sqrt(np.mean(centered**2))),
                "delta_rms": float(np.sqrt(np.mean(delta**2))),
                "foot_dynamic_rms": np.sqrt(np.mean(centered**2, axis=(0, 2, 3))),
                "site_rms": site_rms,
                "site_delta_rms": site_delta_rms,
                "feature": np.r_[site_rms.ravel(), site_delta_rms.ravel()],
            }
        )
    if len(windows) < 10:
        raise ValueError(f"{trial['name']}: too few valid windows ({len(windows)})")
    return windows


def spectrum(trial: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    fs = trial["fs"]
    edge = int(round(fs))
    mask = trial["analysis_valid_row"].copy()
    mask[:edge] = False
    mask[-edge:] = False
    segment = longest_true_slice(mask)
    values = trial["hall"][segment].reshape(-1, 90)
    values = detrend(values, axis=0, type="linear")
    frequency, power = welch(
        values,
        fs=fs,
        nperseg=min(512, len(values)),
        axis=0,
        detrend="constant",
    )
    power = np.mean(power, axis=1)
    band = (frequency >= 0.7) & (frequency <= 3.5)
    band_indices = np.where(band)[0]
    peak_index = int(band_indices[np.argmax(power[band])])
    peak_hz = float(frequency[peak_index])
    prominence = float(power[peak_index] / max(np.median(power[band]), 1.0e-12))
    return frequency, power, peak_hz, prominence


def standardized_centroid_distances(window_data: dict[str, list[dict]]) -> dict:
    features = {
        name: np.stack([row["feature"] for row in rows])
        for name, rows in window_data.items()
    }
    all_features = np.concatenate(list(features.values()), axis=0)
    center = np.median(all_features, axis=0)
    scale = np.quantile(all_features, 0.75, axis=0) - np.quantile(
        all_features, 0.25, axis=0
    )
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    centroids = {
        name: np.median((values - center) / scale, axis=0)
        for name, values in features.items()
    }

    def distance(a: str, b: str) -> float:
        return float(np.linalg.norm(centroids[a] - centroids[b]) / np.sqrt(len(center)))

    return {
        "friction_distance_walkrun": distance("high_walkrun", "low_walkrun"),
        "friction_distance_waist_walk": distance("high_waist", "low_waist"),
        "mode_distance_high_friction": distance("high_walkrun", "high_waist"),
        "mode_distance_low_friction": distance("low_walkrun", "low_waist"),
        "definition": (
            "RMS distance between robust-standardized centroids of 1 s Hall "
            "dynamic/site-delta features; descriptive only, windows are not replicates"
        ),
    }


def write_figure(trials: dict, windows: dict, spectra: dict, site_ratio: dict) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    for name in ORDER:
        rows = windows[name]
        ax_a.plot(
            [row["time_s"] for row in rows],
            [row["dynamic_rms"] for row in rows],
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.3,
            label=DISPLAY[name],
        )
    ax_a.set_xlabel("Trial time (s)")
    ax_a.set_ylabel("1 s detrended Hall RMS (counts)")
    ax_a.set_title("Local magnetic dynamics")
    ax_a.legend(fontsize=6.7, ncol=2, loc="upper left")

    positions = np.arange(len(ORDER))
    for position, name in zip(positions, ORDER):
        values = np.asarray([row["dynamic_rms"] for row in windows[name]])
        parts = ax_b.violinplot(
            values,
            positions=[position],
            widths=0.72,
            showextrema=False,
            showmedians=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[name])
            body.set_edgecolor(COLORS[name])
            body.set_alpha(0.33)
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.2)
        # Deterministic offsets only improve visibility; no observations are sampled.
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax_b.scatter(
            position + jitter,
            values,
            s=7,
            color=COLORS[name],
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
    ax_b.set_xticks(positions, ["High\nwalk–run", "Low\nwalk–run", "High\nwaist", "Low\nwaist"])
    ax_b.set_ylabel("1 s detrended Hall RMS (counts)")
    ax_b.set_title("Window distributions (descriptive, not replicates)")

    for name in ORDER:
        frequency, power, _, _ = spectra[name]
        mask = (frequency >= 0.4) & (frequency <= 4.0)
        ax_c.plot(
            frequency[mask],
            10.0 * np.log10(power[mask] + 1.0e-12),
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.2,
            label=DISPLAY[name],
        )
    ax_c.set_xlabel("Frequency (Hz)")
    ax_c.set_ylabel("Mean Hall PSD (dB counts²/Hz)")
    ax_c.set_title("Temporal spectrum")

    matrix = np.vstack([site_ratio["walkrun"], site_ratio["waist"]])
    image = ax_d.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax_d.axvline(14.5, color="white", linewidth=1.2)
    ax_d.set_yticks([0, 1], ["Walk–run", "Waist-walk"])
    ax_d.set_xticks(
        [0, 4, 9, 14, 15, 19, 24, 29],
        ["L00", "L04", "L09", "L14", "R00", "R04", "R09", "R14"],
        rotation=45,
        ha="right",
    )
    ax_d.set_title("Low/high site dynamics, log₂ ratio")
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.047, pad=0.03)
    colorbar.set_label("log₂(low/high)")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.14,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(
        "Real dual-foot Hall signals: friction response is controller-mode dependent",
        fontsize=10,
        fontweight="bold",
    )
    fig.savefig(
        OUT / "hall_four_condition_overview.png", dpi=400, bbox_inches="tight"
    )
    fig.savefig(OUT / "hall_four_condition_overview.svg", bbox_inches="tight")
    fig.savefig(OUT / "hall_four_condition_overview.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "hall_four_condition_overview.tiff", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = {name: load_trial(name, path) for name, path in TRIALS.items()}
    windows = {name: make_windows(trial) for name, trial in trials.items()}
    spectra = {name: spectrum(trial) for name, trial in trials.items()}

    summaries = []
    site_medians = {}
    for name in ORDER:
        trial = trials[name]
        rows = windows[name]
        dynamic = np.asarray([row["dynamic_rms"] for row in rows])
        delta = np.asarray([row["delta_rms"] for row in rows])
        valid_fraction = float(np.mean(trial["valid_row"]))
        site_medians[name] = np.median(
            np.stack([row["site_rms"] for row in rows]), axis=0
        )
        frequency, power, peak_hz, prominence = spectra[name]
        del frequency, power
        summaries.append(
            {
                "condition": name,
                "surface": trial["metadata"]["surface_label"],
                "mode": trial["metadata"]["controller_mode"],
                "samples": int(len(trial["hall"])),
                "valid_row_fraction": valid_fraction,
                "analysis_retained_fraction": float(
                    np.mean(trial["analysis_valid_row"])
                ),
                "paired_rate_hz": trial["fs"],
                "descriptive_windows_1s": len(rows),
                "window_dynamic_rms_p10": float(np.quantile(dynamic, 0.10)),
                "window_dynamic_rms_median": float(np.median(dynamic)),
                "window_dynamic_rms_p90": float(np.quantile(dynamic, 0.90)),
                "window_delta_rms_median": float(np.median(delta)),
                "spectral_peak_0p7_3p5_hz": peak_hz,
                "spectral_peak_prominence": prominence,
                "temperature_left_c": float(
                    np.mean(trial["temperature"][:, 0]) / 10.0
                ),
                "temperature_right_c": float(
                    np.mean(trial["temperature"][:, 1]) / 10.0
                ),
            }
        )

    site_ratio = {
        "walkrun": np.log2(
            np.maximum(site_medians["low_walkrun"], 1.0e-9)
            / np.maximum(site_medians["high_walkrun"], 1.0e-9)
        ).ravel(),
        "waist": np.log2(
            np.maximum(site_medians["low_waist"], 1.0e-9)
            / np.maximum(site_medians["high_waist"], 1.0e-9)
        ).ravel(),
    }
    distances = standardized_centroid_distances(windows)
    result = {
        "format": "real-g1-hall-four-condition-analysis-v1",
        "measurement_boundary": "raw Bx/By/Bz only; no force conversion",
        "sample_independence_warning": (
            "one physical trial per condition; 1 s windows are autocorrelated "
            "descriptive samples and are not independent replicates"
        ),
        "exclusion_rule": (
            "exclude first/last 1 s and any row with foot transport invalid, "
            "temperature sentinel <= -9000, or all-zero Hall site, plus a "
            "1 s guard band before/after invalid rows to reject reconnect transients"
        ),
        "conditions": summaries,
        "centroid_distances": distances,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "condition_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (OUT / "window_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("condition", "time_s", "dynamic_rms", "delta_rms"),
        )
        writer.writeheader()
        for name in ORDER:
            for row in windows[name]:
                writer.writerow(
                    {
                        "condition": name,
                        "time_s": row["time_s"],
                        "dynamic_rms": row["dynamic_rms"],
                        "delta_rms": row["delta_rms"],
                    }
                )
    write_figure(trials, windows, spectra, site_ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
