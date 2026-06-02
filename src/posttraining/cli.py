import argparse


def validate_step_budget(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Keep the explicit stage budgets aligned with the advertised
    # total budget so command-line mistakes fail before training.
    # ---------------------------------------------------------
    if args.magpie_steps + args.everyday_steps != args.max_steps:
        raise ValueError("magpie_steps plus everyday_steps must equal max_steps")
