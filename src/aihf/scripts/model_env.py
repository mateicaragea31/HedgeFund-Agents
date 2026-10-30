"""Print an experiment's model settings as shell variables, for the sbatch scripts.

    eval "$(python -m aihf.scripts.model_env configs/experiments/e1_llama70b.yaml)"

sets MODEL_CONFIG, MODEL_DIR, SERVED_NAME, TP, PP, GPUS, MAX_MODEL_LEN, CHAT_TEMPLATE, so the job
serves exactly the model the experiment names (one source of truth: the yaml files).
"""

import shlex
import sys
from pathlib import Path

from aihf.config import load_experiment


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m aihf.scripts.model_env <experiment.yaml>", file=sys.stderr)
        return 2
    experiment, model = load_experiment(Path(args[0]))
    if "${" in model.path:
        print(f"model path not filled in (is AIHF_LAB set?): {model.path}", file=sys.stderr)
        return 2
    tp = int(model.vllm.get("tensor_parallel_size", 1))
    pp = int(model.vllm.get("pipeline_parallel_size", 1))
    values = {
        "MODEL_CONFIG": str(experiment.model_profile),
        "MODEL_DIR": model.path,
        "SERVED_NAME": model.name,
        "TP": tp, "PP": pp, "GPUS": tp * pp,
        "MAX_MODEL_LEN": model.vllm.get("max_model_len", 65536),
        "CHAT_TEMPLATE": model.vllm.get("chat_template", ""),
        "REPEATS": experiment.repeats,
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
