import argparse
import json
from pathlib import Path

from .config import Config
from .experiment import run
from .visualization import load_result, render


def main():
    parser = argparse.ArgumentParser(description="V6 warehouse validation and visualization")
    parser.add_argument("--mode", choices=["hybrid", "centralized", "both"], default="both")
    parser.add_argument("--horizon", type=float, default=60.)
    parser.add_argument("--robots", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--packet-loss", type=float, default=.10)
    parser.add_argument("--latency-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/validation"))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    modes = ["centralized", "hybrid"] if args.mode == "both" else [args.mode]
    config = Config(robots=args.robots, horizon=args.horizon, seed=args.seed,
                    packet_loss=args.packet_loss, latency_steps=args.latency_steps)
    results = []
    for mode in modes:
        result = load_result(args.output, mode) if args.render_only else run(config, mode, args.output)
        results.append(result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    render(results, args.output, make_gif=not args.no_gif)
    print(f"Viewer: {args.output.resolve() / 'index.html'}")


if __name__ == "__main__":
    main()

