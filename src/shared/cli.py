import argparse


def require(
    condition: bool,
    parser: argparse.ArgumentParser,
    message: str,
) -> None:
    # ---------------------------------------------------------
    # Stop CLI parsing with argparse's standard error format when
    # a parsed argument combination is invalid.
    # ---------------------------------------------------------
    if not condition:
        parser.error(message)
