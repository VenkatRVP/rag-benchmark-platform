import re
def normalize_unicode_spaces(text: str) -> str:
    return re.sub(r'[^\S\n]+', ' ', text).strip()