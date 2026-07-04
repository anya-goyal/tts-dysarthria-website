#!/usr/bin/env python3
"""
Copy audio files for a given sample across experiment types.
CSVs are pre-pruned so there is exactly one row per experiment config.
Also copies orig.wav and writes info.txt into each destination folder.

Usage:
    # Single experiment
    python copy_best_audios.py <experiment> <sample_name> [--shift SHIFT] [--model MODEL]

    # All experiments at once
    python copy_best_audios.py --all <sample_name>

Experiments:
    basic_inf_reran   - one row per (sample_name, model, shift)
    loop_reran        - one row per (sample_name, loop_iteration)
    local             - one row per (sample_name, steering_strength)
    global            - same as local but different source/dest paths

--all runs every combination of:
    shifts:  in_order, shuffled, shifted
    models:  cosyvoice, f5-tts, voicecraft, voicestar
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded CSV paths
# ---------------------------------------------------------------------------

CSV_PATHS = {
    "basic_inf_reran": "/saltpool0/data/anyagoyal/paper_results/basic_inf_reran/pruned_all_results_master.csv",
    "loop_reran":      "/saltpool0/data/anyagoyal/paper_results/loop_reran/pruned_all_results_master.csv",
    "local":           "/saltpool0/data/anyagoyal/paper_results/steering/local/{shift}/pruned_all_results_master.csv",
    "global":          "/saltpool0/data/anyagoyal/paper_results/steering/global/{shift}/pruned_all_results_master.csv",
}

BASIC_INF_SHIFTS  = ["in_order", "shifted"]
STEERING_SHIFTS   = ["in_order", "shuffled"]
ALL_MODELS        = ["cosyvoice", "f5-tts", "voicecraft", "voicestar"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_strength(val: float) -> str:
    """Format a steering strength float with no unnecessary trailing zeros."""
    s = f"{val:f}".rstrip("0").rstrip(".")
    return s if s else "0"


def iter_col(df: pd.DataFrame) -> str:
    """Return whichever iteration column exists in the dataframe."""
    for col in ("gen_iteration", "iteration"):
        if col in df.columns:
            return col
    raise ValueError("Neither 'gen_iteration' nor 'iteration' column found in CSV.")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  [WARN] Source not found, skipping: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied: {src}\n      --> {dst}")


def copy_orig_and_info(sample_name: str, dest_dir: Path) -> None:
    """Copy orig.wav and write info.txt into dest_dir."""
    orig_src = Path(f"/saltpool0/data/anyagoyal/torgo/{sample_name}.wav")
    copy_file(orig_src, dest_dir / "orig.wav")

    transcript_src = Path(f"/saltpool0/data/anyagoyal/torgo/{sample_name}.txt")
    if not transcript_src.exists():
        print(f"  [WARN] Transcript not found, skipping info.txt: {transcript_src}")
        return
    transcript = transcript_src.read_text().strip()
    dest_dir.mkdir(parents=True, exist_ok=True)
    info_path = dest_dir / "info.txt"
    info_path.write_text(f"Sample: {sample_name}\nTranscript: {transcript}\n")
    print(f"  Wrote:  {info_path}")


# ---------------------------------------------------------------------------
# Experiment handlers
# ---------------------------------------------------------------------------

def handle_basic_inf(df: pd.DataFrame, sample_name: str, shift: str, model: str) -> None:
    mask = (df["sample_name"] == sample_name)
    if "model" in df.columns:
        mask &= (df["model"] == model)
    if "shift" in df.columns:
        mask &= (df["shift"] == shift)
    df = df[mask].copy()

    if df.empty:
        print(f"  [WARN] basic_inf_reran: no rows for model={model} shift={shift}, skipping.")
        return

    it_col = iter_col(df)
    row = df.iloc[0]
    iteration = int(row[it_col])

    dest_dir = Path(
        f"/saltpool0/data/anyagoyal/paper_results/demo_audios"
        f"/basic_inf/{model}/{shift}/{sample_name}"
    )
    src = Path(
        f"/saltpool0/data/anyagoyal/generated_audios_basic_inference"
        f"/{model}/{shift}/{sample_name}/gen_{iteration}.wav"
    )
    copy_file(src, dest_dir / "gen.wav")
    copy_orig_and_info(sample_name, dest_dir)


def handle_loop(df: pd.DataFrame, sample_name: str) -> None:
    df = df[df["sample_name"] == sample_name].copy()

    if df.empty:
        print(f"  [WARN] loop_reran: no rows for sample={sample_name}, skipping.")
        return

    if "loop_iteration" not in df.columns:
        print("[ERROR] 'loop_iteration' column not found in CSV.")
        return

    df = df.dropna(subset=["gen_iteration"])

    if df.empty:
        print(f"  [WARN] loop_reran: no rows with gen_iteration for sample={sample_name}, skipping.")
        return

    dest_sample_dir = Path(
        f"/saltpool0/data/anyagoyal/paper_results/demo_audios/loop/{sample_name}"
    )

    for _, row in df.iterrows():
        loop_iter = int(row["loop_iteration"])
        gen_iter = int(row["gen_iteration"])

        src = Path(
            f"/saltpool0/data/anyagoyal/loop_generated_audios/torgo/iterative_paper"
            f"/loop_iter_{loop_iter}/{sample_name}/gen_{gen_iter}.wav"
        )
        copy_file(src, dest_sample_dir / f"loop_iter_{loop_iter}.wav")

    copy_orig_and_info(sample_name, dest_sample_dir)


def handle_steering(df: pd.DataFrame, sample_name: str, shift: str, mode: str) -> None:
    df = df[df["sample_name"] == sample_name].copy()

    if df.empty:
        print(f"  [WARN] {mode}: no rows for sample={sample_name} shift={shift}, skipping.")
        return

    if "steering_strength" not in df.columns:
        print("[ERROR] 'steering_strength' column not found in CSV.")
        return

    speaker = sample_name.split("_")[0]
    shift_mod = f"{'indiv' if mode == 'local' else 'loso'}_{shift}-{speaker}"

    dest_sample_dir = Path(
        f"/saltpool0/data/anyagoyal/paper_results/demo_audios"
        f"/{mode}/{shift}/{sample_name}"
    )

    for _, row in df.iterrows():
        strength_fmt = fmt_strength(float(row["steering_strength"]))

        src = Path(
            f"/saltpool0/data/anyagoyal/steering_generated_audios/torgo"
            f"/{shift_mod}/f5-tts/{strength_fmt}/{sample_name}/gen_0.wav"
        )
        copy_file(src, dest_sample_dir / f"strength_{strength_fmt}.wav")

    copy_orig_and_info(sample_name, dest_sample_dir)


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all(sample_name: str) -> None:
    # basic_inf_reran: every model x in_order + shifted
    print("\n=== basic_inf_reran ===")
    df_basic = pd.read_csv(CSV_PATHS["basic_inf_reran"])
    for model in ALL_MODELS:
        for shift in BASIC_INF_SHIFTS:
            print(f"\n  model={model}  shift={shift}")
            handle_basic_inf(df_basic, sample_name, shift, model)

    # loop_reran: no shift/model axis
    print("\n=== loop_reran ===")
    df_loop = pd.read_csv(CSV_PATHS["loop_reran"])
    handle_loop(df_loop, sample_name)

    # local + global: in_order + shuffled
    for mode in ("local", "global"):
        print(f"\n=== {mode} ===")
        for shift in STEERING_SHIFTS:
            print(f"\n  shift={shift}")
            csv_path = CSV_PATHS[mode].format(shift=shift)
            df_steer = pd.read_csv(csv_path)
            handle_steering(df_steer, sample_name, shift, mode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy audio files for a given sample."
    )
    parser.add_argument("sample_name", help="Sample name, e.g. F03_S3_array_0038")
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments x all shifts x all models.",
    )
    parser.add_argument(
        "experiment", nargs="?",
        choices=["basic_inf_reran", "loop_reran", "local", "global"],
        help="Which experiment to process (omit when using --all).",
    )
    parser.add_argument(
        "--shift",
        default=None,
        help="Shift value (required for basic_inf_reran, local, global).",
    )
    parser.add_argument(
        "--model",
        default="f5-tts",
        help="Model name for basic_inf_reran (default: f5-tts).",
    )
    args = parser.parse_args()

    sample_name: str = args.sample_name

    if args.all:
        run_all(sample_name)
        return

    if not args.experiment:
        parser.error("experiment is required unless --all is passed.")

    experiment: str = args.experiment
    shift: str | None = args.shift
    model: str = args.model

    if experiment in ("basic_inf_reran", "local", "global") and shift is None:
        parser.error(f"--shift is required for experiment '{experiment}'")

    csv_path = CSV_PATHS[experiment].format(shift=shift)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows.")

    if experiment == "basic_inf_reran":
        handle_basic_inf(df, sample_name, shift, model)
    elif experiment == "loop_reran":
        handle_loop(df, sample_name)
    elif experiment in ("local", "global"):
        handle_steering(df, sample_name, shift, mode=experiment)


if __name__ == "__main__":
    main()