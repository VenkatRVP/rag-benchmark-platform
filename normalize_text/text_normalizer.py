from normalize_text.collapse_blank_lines import collapse_blank_lines
from normalize_text.collapse_multiple_spaces import collapse_multiple_spaces
from normalize_text.normalize_line_endings import normalize_line_endings
from normalize_text.normalize_unicode_spaces import normalize_unicode_spaces
from normalize_text.remove_control_characters import remove_control_characters
from normalize_text.trim_lines import trim_lines


def normalize(text: str) -> str:
    text = normalize_line_endings(text)
    text = remove_control_characters(text)
    text = normalize_unicode_spaces(text)
    text = collapse_multiple_spaces(text)
    text = trim_lines(text)
    text = collapse_blank_lines(text)

    return text