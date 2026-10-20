import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "2")
import argparse
from .contracts import read_json, load_dataset
from .conditions import prepare
from .runner import reconstruct, evaluate
from .summary import summarize


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark with user-supplied input and reconstruction adapters")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ["validate-input", "prepare"]:
        command = commands.add_parser(name)
        command.add_argument("dataset")
        command.add_argument("--config", default="benchmark.json")
        if name == "prepare":
            command.add_argument("--output", required=True)
    command = commands.add_parser("run")
    command.add_argument("run_directory")
    command.add_argument("command", nargs=argparse.REMAINDER)
    command = commands.add_parser("evaluate")
    command.add_argument("run_directory")
    command = commands.add_parser("summarize")
    command.add_argument("run_directory")
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "validate-input":
        scans = load_dataset(args.dataset, read_json(args.config))
        print(f"Validated {len(scans)} scan(s); full protocol requires 9 path/speed pairs")
    elif args.action == "prepare":
        print(f"Prepared {len(prepare(args.dataset, read_json(args.config), args.output))} cases")
    elif args.action == "run":
        reconstruct(args.run_directory, args.command[1:] if args.command[:1] == ["--"] else args.command)
    elif args.action == "evaluate":
        evaluate(args.run_directory)
    else:
        summarize(args.run_directory, args.output)


if __name__ == "__main__":
    main()
