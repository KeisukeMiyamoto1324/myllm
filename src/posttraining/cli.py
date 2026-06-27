import argparse


def validate_repeat_epochs(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Reject invalid epoch counts before loading the model or
    # materializing the SFT dataset.
    # ---------------------------------------------------------
    if args.repeat_epochs <= 0:
        raise ValueError("repeat_epochs must be positive")
