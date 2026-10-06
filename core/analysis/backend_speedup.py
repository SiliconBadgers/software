"""Build explicit accelerated-backend speedups against each identified CPU dataset."""
import csv

import datasets
from policy import timing_eligible
from result_io import open_csv, write_json


def compare(summaries, dataset_rows, accelerated="cuda", experiment=None):
    accelerated_rows = {row["prompt_tokens"]: row for row in summaries
                        if datasets.for_run(dataset_rows, row["run"])
                        and datasets.for_run(dataset_rows, row["run"])["name"] == accelerated}
    baselines = [row for row in dataset_rows if row.get("role") in {
        "cpu_reference", "cuda_build_cpu_control", "cuda_build_cpu_no_host_control"}]
    output = []
    for baseline in baselines:
        for row in summaries:
            dataset = datasets.for_run(dataset_rows, row["run"])
            if not dataset or dataset["name"] != baseline["name"] or row["prompt_tokens"] not in accelerated_rows:
                continue
            fast = accelerated_rows[row["prompt_tokens"]]
            output.append({
                "reference_dataset": baseline["name"], "reference_role": baseline["role"],
                "accelerated_dataset": accelerated, "prompt_tokens": row["prompt_tokens"],
                "timing_eligible_for_report": timing_eligible(experiment, row["prompt_tokens"]) if experiment else None,
                "prefill_speedup": row["prefill_seconds"] / fast["prefill_seconds"],
                "decode_speedup": row["decode_ms"] / fast["decode_ms"],
            })
    return output


def write_outputs(out_dir, rows):
    write_json(out_dir / "backend-speedups.json", rows)
    if rows:
        with open_csv(out_dir / "backend-speedups.csv") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
