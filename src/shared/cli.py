def require(condition: bool, message: str) -> None:
    # ---------------------------------------------------------
    # Raise one consistent CLI validation error that each parser
    # can convert into argparse's standard error output.
    # ---------------------------------------------------------
    if not condition:
        raise ValueError(message)
