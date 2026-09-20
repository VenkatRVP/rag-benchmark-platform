import unicodedata


def remove_control_characters(text: str) -> str:

    white_list = {"\n", "\t", "\r"}

    return "".join(
        c
        for c in text
        if unicodedata.category(c) != "Cc"
        or c in white_list
    )