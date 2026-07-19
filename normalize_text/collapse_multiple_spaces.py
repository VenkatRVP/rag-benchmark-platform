import re
def collapse_multiple_spaces(text: str)->str:
    return re.compile(r' {2,}').sub(' ',text)