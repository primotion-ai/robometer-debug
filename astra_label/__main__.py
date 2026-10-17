import argparse
import json
from pathlib import Path
import sys

from .pipeline import run
from .connection import add_connection_arguments, connection_overrides


def main():
    parser = argparse.ArgumentParser(description="Reference-keyframe conditioned GPT-6 Astra progress annotation")
    parser.add_argument("--config", type=Path, required=True, help="JSON job config")
    parser.add_argument("--output", type=Path, required=True, help="new output directory, or same run for resume")
    parser.add_argument("--prepare-only", action="store_true", help="validate data and export images/prompts; no model call")
    parser.add_argument("--model", help="explicit model override; never changed automatically")
    parser.add_argument("--reasoning", choices=("low", "medium", "high", "xhigh", "max"))
    add_connection_arguments(parser)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        config.update(connection_overrides(args))
        for key in ("model", "reasoning"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        run(config, args.output, args.prepare_only)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'Prepared' if args.prepare_only else 'Saved labels'}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
