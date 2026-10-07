import re


def clean_plate_text(text):

    if not text:
        return ""

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    replacements = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return text