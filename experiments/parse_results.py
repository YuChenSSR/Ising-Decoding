#!/usr/bin/env python3
"""Parse experiment logs and extract metrics into CSV files for plotting."""
import re
import os
import csv
import glob

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "parsed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_log(filepath):
    """Extract key metrics from a single inference log file."""
    with open(filepath, "r") as f:
        text = f.read()

    metrics = {"file": os.path.basename(filepath)}

    # LER
    m = re.search(r"LER - Avg:\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        metrics["ler_baseline"] = float(m.group(1))
        metrics["ler_predecoder"] = float(m.group(2))

    # PyMatching latency
    m = re.search(r"PyMatching latency - Avg \(µs/round\):\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        metrics["pm_latency_baseline_us"] = float(m.group(1))
        metrics["pm_latency_predecoder_us"] = float(m.group(2))

    # PyMatching speedup
    m = re.search(r"PyMatching speedup.*?:\s+([\d.]+)x", text)
    if m:
        metrics["pm_speedup"] = float(m.group(1))

    # Phase timing breakdown
    m = re.search(r"Model forward:\s+([\d.]+)s", text)
    if m:
        metrics["model_forward_s"] = float(m.group(1))

    m = re.search(r"PyMatching baseline:\s+([\d.]+)s", text)
    if m:
        metrics["pm_baseline_total_s"] = float(m.group(1))

    m = re.search(r"PyMatching predecoder:\s+([\d.]+)s", text)
    if m:
        metrics["pm_predecoder_total_s"] = float(m.group(1))

    m = re.search(r"GPU→CPU transfer:\s+([\d.]+)s", text)
    if m:
        metrics["gpu_cpu_transfer_s"] = float(m.group(1))

    m = re.search(r"Data generation:\s+([\d.]+)s", text)
    if m:
        metrics["data_gen_s"] = float(m.group(1))

    # TRT engine build time
    m = re.search(r"TensorRT engine built in ([\d.]+)s", text)
    if m:
        metrics["trt_build_s"] = float(m.group(1))

    # TRT first batch
    m = re.search(r"TensorRT first batch executed in ([\d.]+)s", text)
    if m:
        metrics["trt_first_batch_s"] = float(m.group(1))

    # TRT layer precisions
    m = re.search(r"TensorRT engine layer precisions: (\{.*?\})", text)
    if m:
        metrics["trt_precisions"] = m.group(1)

    # Syndrome density
    m = re.search(r"Density reduction:\s+([\d.]+)%", text)
    if m:
        metrics["syndrome_density_reduction_pct"] = float(m.group(1))

    # Total samples decoded
    m = re.search(r"Total rounds decoded:\s+([\d,]+)", text)
    if m:
        metrics["total_rounds"] = int(m.group(1).replace(",", ""))

    # ONNX workflow
    m = re.search(r"ONNX_WORKFLOW=(\S+)", text)
    if m:
        metrics["onnx_workflow"] = m.group(1)

    # Model params
    m = re.search(r"Model loaded \(([\d,]+) parameters\)", text)
    if m:
        metrics["model_params"] = int(m.group(1).replace(",", ""))

    return metrics


def main():
    log_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.log")))
    if not log_files:
        print(f"No log files found in {RESULTS_DIR}")
        return

    all_metrics = []
    for lf in log_files:
        m = parse_log(lf)
        all_metrics.append(m)
        print(f"Parsed: {m['file']}")

    # Collect all keys
    all_keys = []
    for m in all_metrics:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    # Write combined CSV
    csv_path = os.path.join(OUTPUT_DIR, "all_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nWritten: {csv_path}")

    # Experiment-specific CSVs
    for prefix in ["exp1", "exp2", "exp3", "exp4"]:
        subset = [m for m in all_metrics if m["file"].startswith(prefix)]
        if not subset:
            continue
        keys = []
        for m in subset:
            for k in m:
                if k not in keys:
                    keys.append(k)
        csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(subset)
        print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
