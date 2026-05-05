import argparse


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for two-stage SFT from a pretrained
    # base model into a chat-oriented model artifact.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="models/chat-model")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--magpie-steps", type=int, default=11000)
    parser.add_argument("--everyday-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=50)
    return parser.parse_args()


def validate_step_budget(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Keep the explicit stage budgets aligned with the advertised
    # total budget so command-line mistakes fail before training.
    # ---------------------------------------------------------
    if args.magpie_steps + args.everyday_steps != args.max_steps:
        raise ValueError("magpie_steps plus everyday_steps must equal max_steps")
