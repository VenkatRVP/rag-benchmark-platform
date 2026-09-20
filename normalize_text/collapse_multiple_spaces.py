import re


def collapse_multiple_spaces(text: str) -> str:

    return re.sub(
        r" {2,}",
        " ",
        text
    )