"""Summarize raw uninstrumented and diagnostic phase measurements."""
import csv
import json
import statistics

from result_io import exists, open_result, open_csv, parse_paths, write_json


def summarize(results_dir, experiment):
    summaries = []
    for name in experiment["policy"]["summary_runs"]:
        path = results_dir / (name + ".jsonl")
        if not exists(path):
            continue
        with open_result(path) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        rows = [r for r in rows if r["kind"] == "measurement"]
        for n in sorted(set(r["prompt_tokens"] for r in rows)):
            rs = [r for r in rows if r["prompt_tokens"] == n]
            pre = [r["prefill_us"] / 1e6 for r in rs]
            dec = [statistics.mean(r["decode_us"]) / 1e3 for r in rs]
            summaries.append({
                "run": name, "prompt_tokens": n, "repetitions": len(rs),
                "decode_steps_per_rep": len(rs[0]["decode_us"]),
                "prefill_seconds": statistics.median(pre), "prefill_tps": n / statistics.median(pre),
                "decode_ms": statistics.median(dec), "decode_tps": 1000 / statistics.median(dec),
                "prefill_min_s": min(pre), "prefill_max_s": max(pre),
                "decode_min_ms": min(dec), "decode_max_ms": max(dec),
                "prefill_samples_seconds": pre, "decode_run_means_ms": dec})
    if not summaries:
        raise ValueError("No measurement files found in " + str(results_dir))
    return summaries


def write_outputs(out_dir, summaries):
    write_json(out_dir / "summary.json", summaries)
    with open_csv(out_dir / "summary.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main(argv=None):
    args = parse_paths(__doc__, argv=argv)
    summaries = summarize(args.results_dir, args.experiment)
    write_outputs(args.output_dir, summaries)
    print(json.dumps(summaries, indent=2))
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
