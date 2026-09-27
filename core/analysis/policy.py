"""Experiment definitions (core/experiments/*.json) and the rules derived from them."""
import json
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent
DEFAULT_EXPERIMENT = "qwen35-2b-q4km"


def load_experiment(name_or_path=DEFAULT_EXPERIMENT):
    path = Path(name_or_path)
    if not path.is_file():
        path = CORE / "experiments" / (str(name_or_path) + ".json")
    return json.loads(path.read_text(encoding="utf-8"))


def timing_eligible(experiment, prompt_tokens):
    return prompt_tokens not in experiment["policy"]["timing_excluded_prompt_tokens"]


def is_diagnostic(experiment, prompt_tokens):
    return prompt_tokens in experiment["policy"]["diagnostic_prompt_tokens"]


def case_prefix(experiment, backend, prompt_tokens):
    """Baseline file prefix: 'cpu-baseline', or 'cpu-8k' for lengths run as separate cases."""
    tag = experiment["policy"]["separate_case_prefixes"].get(str(prompt_tokens))
    return f"{backend}-{tag}" if tag else f"{backend}-baseline"


def profile_prefixes(experiment, prompt_tokens):
    table = experiment["policy"]["profile_prefixes"]
    return table.get(str(prompt_tokens), table["default"])
