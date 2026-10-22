"""Print a symbolic compiler plan; unsupported work fails unless explicitly inventoried."""
import argparse
import sys

from cdfg import Graph
from .lower import compile_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="captured phase JSON or JSON.gz")
    parser.add_argument("--phase", required=True, choices=("prefill", "decode"))
    parser.add_argument("--tensor-id", type=int, help="select one active result tensor ID")
    parser.add_argument("--allow-partial", action="store_true",
                        help="inventory unsupported nodes with exit 0; does not enable execution")
    args = parser.parse_args(argv)
    try:
        plan = compile_graph(Graph(args.capture, args.phase), tensor_id=args.tensor_id)
        output = plan.to_json()
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(output)
    if plan.diagnostics:
        print(f"{len(plan.diagnostics)} active operations remain unsupported; plan is symbolic only",
              file=sys.stderr)
        return 0 if args.allow_partial else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
