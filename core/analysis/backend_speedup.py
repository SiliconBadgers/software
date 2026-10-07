"""Build explicit accelerated-backend speedups against each identified CPU dataset."""
import csv

import datasets
from policy import timing_eligible
from result_io import open_csv, write_json

FIELDS = ["reference_dataset", "reference_role", "accelerated_dataset", "prompt_tokens",
          "timing_eligible_for_report", "prefill_speedup", "decode_speedup"]


def compare(summaries, dataset_rows, accelerated="cuda", experiment=None, *, include_controls=False):
    accelerated_rows = {}
    for row in summaries:
        dataset = datasets.for_run(dataset_rows, row["run"])
        if (dataset and dataset["name"] == accelerated
                and dataset.get("role") == "accelerated_execution" and not datasets.is_fallback(dataset)):
            accelerated_rows[row["prompt_tokens"]] = row
    roles = {"cpu_reference"}
    if include_controls:
        roles.update({"cuda_build_cpu_control", "cuda_build_cpu_no_host_control"})
    baselines = [row for row in dataset_rows if row.get("role") in roles and not datasets.is_fallback(row)]
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
    # Always rewrite CSV too: reanalysis must not leave an earlier invalid speedup behind.
    with open_csv(out_dir / "backend-speedups.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
