import re


def collapse_blank_lines(text: str) -> str:

    return re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )