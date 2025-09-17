import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RunMetrics:
    model_id: str
    setting: Optional[str]
    mse: Optional[float]
    mae: Optional[float]
    total_gen_time_s: Optional[float]
    per_batch_s: Optional[float]
    accepted: Optional[int]
    attempted: Optional[int]
    acceptance_pct: Optional[float]
    tol: Optional[float]
    bias: Optional[float]
    adapt_c: Optional[int]


# NOTE: keep typing simple for runtime safety on older Python versions


RUN_HEADER_RE = re.compile(r"^\[RUN\]\s+(?P<model_id>.+)\s*$")
SETTING_RE = re.compile(r"^setting:(?P<setting>.*)$")
MSE_MAE_RE = re.compile(r"^mse:(?P<mse>[-+]?\d*\.\d+|\d+),\s*mae:(?P<mae>[-+]?\d*\.\d+|\d+)\s*$")
TIME_RE = re.compile(
    r"^total_gen_time_s:(?P<total>[-+]?\d*\.\d+|\d+),\s*per_batch_s:(?P<per>[-+]?\d*\.\d+|\d+)\s*$"
)
ACCEPT_RE = re.compile(
    r"^accepted:(?P<acc>\d+),\s*attempted:(?P<attempt>\d+),\s*acceptance_pct:(?P<pct>[-+]?\d*\.\d+|\d+)\s*$"
)

# Example model id patterns we want to support:
# ETTh1_specK3_adapt_c4_tol0p25_bias1p5
# ETTh1_specK3_adapt_c4
# ETTh1_specK3_fixed_s1p0
TOL_RE = re.compile(r"tol(?P<tol>[0-9]+p[0-9]+)")
BIAS_RE = re.compile(r"bias(?P<bias>[0-9]+p[0-9]+)")
ADAPT_C_RE = re.compile(r"adapt_c(?P<c>\d+)")


def parse_p_notation(value: str) -> Optional[float]:
    # Convert 0p25 -> 0.25, 1p5 -> 1.5
    if not value:
        return None
    return float(value.replace("p", "."))


def extract_params_from_model_id(model_id: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    tol_match = TOL_RE.search(model_id)
    bias_match = BIAS_RE.search(model_id)
    c_match = ADAPT_C_RE.search(model_id)
    tol = parse_p_notation(tol_match.group("tol")) if tol_match else None
    bias = parse_p_notation(bias_match.group("bias")) if bias_match else None
    adapt_c = int(c_match.group("c")) if c_match else None
    return tol, bias, adapt_c


def parse_summary_file(path: str) -> List[RunMetrics]:
    results: List[RunMetrics] = []
    current: Optional[RunMetrics] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            run_header = RUN_HEADER_RE.match(line)
            if run_header:
                # Flush previous
                if current is not None:
                    results.append(current)
                model_id = run_header.group("model_id").strip()
                tol, bias, adapt_c = extract_params_from_model_id(model_id)
                current = RunMetrics(
                    model_id=model_id,
                    setting=None,
                    mse=None,
                    mae=None,
                    total_gen_time_s=None,
                    per_batch_s=None,
                    accepted=None,
                    attempted=None,
                    acceptance_pct=None,
                    tol=tol,
                    bias=bias,
                    adapt_c=adapt_c,
                )
                continue

            if current is None:
                continue

            m_setting = SETTING_RE.match(line)
            if m_setting:
                current.setting = m_setting.group("setting").strip()
                continue

            m_mse = MSE_MAE_RE.match(line)
            if m_mse:
                current.mse = float(m_mse.group("mse"))
                current.mae = float(m_mse.group("mae"))
                continue

            m_time = TIME_RE.match(line)
            if m_time:
                current.total_gen_time_s = float(m_time.group("total"))
                current.per_batch_s = float(m_time.group("per"))
                continue

            m_acc = ACCEPT_RE.match(line)
            if m_acc:
                current.accepted = int(m_acc.group("acc"))
                current.attempted = int(m_acc.group("attempt"))
                current.acceptance_pct = float(m_acc.group("pct"))
                continue

        # Flush last
        if current is not None:
            results.append(current)

    return results


def compute_baselines(rows: List[RunMetrics]) -> Dict[str, float]:
    # Baseline per setting: use the per_batch_s of acceptance == 0 if available
    baselines: Dict[str, float] = {}
    for r in rows:
        if r.setting and r.acceptance_pct is not None and r.acceptance_pct == 0.0 and r.per_batch_s:
            if r.setting not in baselines:
                baselines[r.setting] = r.per_batch_s
            else:
                # Keep the earliest one; or choose min
                baselines[r.setting] = min(baselines[r.setting], r.per_batch_s)
    return baselines


def maybe_plot(csv_path: str, rows: List[RunMetrics]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    xs = []  # acceptance
    ys = []  # per_batch_s
    labels = []
    colors = []
    for r in rows:
        if r.acceptance_pct is None or r.per_batch_s is None:
            continue
        xs.append(r.acceptance_pct)
        ys.append(r.per_batch_s)
        label = r.model_id
        labels.append(label)
        # Color by tol where available
        if r.tol is None:
            colors.append("gray")
        elif r.tol <= 0.15:
            colors.append("tab:blue")
        elif r.tol <= 0.20:
            colors.append("tab:orange")
        else:
            colors.append("tab:green")

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, c=colors, s=50, edgecolors="k")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Acceptance %")
    plt.ylabel("Per-batch time (s)")
    plt.title("Acceptance vs Speed (lower is faster)")
    plt.grid(True, linestyle=":", linewidth=0.5)
    out_path = os.path.splitext(csv_path)[0] + "_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result_inference_summary.txt")
    parser.add_argument("--output", default="tol_bias_sweep_metrics.csv")
    parser.add_argument("--filter_contains", default="tol", help="Only export runs whose model_id contains this substring (empty for all)")
    parser.add_argument("--plot", default="auto", help="'auto' to attempt plot, 'off' to skip, or provide explicit png path base via output name")
    args = parser.parse_args()

    rows = parse_summary_file(args.input)

    # Filter to tol/bias sweep by default (contains 'tol')
    if args.filter_contains:
        rows = [r for r in rows if args.filter_contains in (r.model_id or "")]  # type: ignore

    # Compute baselines by setting
    baselines = compute_baselines(parse_summary_file(args.input))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "model_id",
        "setting",
        "tol",
        "bias",
        "adapt_c",
        "mse",
        "mae",
        "total_gen_time_s",
        "per_batch_s",
        "accepted",
        "attempted",
        "acceptance_pct",
        "baseline_per_batch_s",
        "speedup_vs_baseline",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            baseline_per = baselines.get(r.setting or "", None)
            speedup = None
            if baseline_per and r.per_batch_s:
                try:
                    speedup = baseline_per / r.per_batch_s
                except ZeroDivisionError:
                    speedup = None

            writer.writerow(
                {
                    "model_id": r.model_id,
                    "setting": r.setting or "",
                    "tol": r.tol if r.tol is not None else "",
                    "bias": r.bias if r.bias is not None else "",
                    "adapt_c": r.adapt_c if r.adapt_c is not None else "",
                    "mse": r.mse if r.mse is not None else "",
                    "mae": r.mae if r.mae is not None else "",
                    "total_gen_time_s": r.total_gen_time_s if r.total_gen_time_s is not None else "",
                    "per_batch_s": r.per_batch_s if r.per_batch_s is not None else "",
                    "accepted": r.accepted if r.accepted is not None else "",
                    "attempted": r.attempted if r.attempted is not None else "",
                    "acceptance_pct": r.acceptance_pct if r.acceptance_pct is not None else "",
                    "baseline_per_batch_s": baseline_per if baseline_per is not None else "",
                    "speedup_vs_baseline": speedup if speedup is not None else "",
                }
            )

    print(f"Wrote CSV: {os.path.abspath(args.output)}")

    plot_path = None
    if args.plot != "off":
        plot_path = maybe_plot(args.output, rows)
        if plot_path:
            print(f"Wrote plot: {plot_path}")
        else:
            print("Plotting skipped (matplotlib not available)")


if __name__ == "__main__":
    main()


