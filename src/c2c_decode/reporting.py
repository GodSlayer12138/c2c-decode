"""Write separate English and Chinese reports without changing measurements."""
import json
import re
from importlib.resources import files
from pathlib import Path


def chinese_markdown(text):
    translations = json.loads(files("c2c_decode").joinpath(
        "report_translations.json").read_text(encoding="utf-8"))
    output = []
    for line in text.split("\n"):
        if line.startswith("|"):
            cells = line.split("|")
            output.append("|".join(
                f" {translations[cell.strip()]} "
                if cell.strip() in translations else cell for cell in cells))
            continue
        heading, body = re.fullmatch(r"(#{1,6} )?(.*)", line).groups()
        translated = translations.get(body)
        output.append(line if translated is None else
                      f"{heading}{translated}" if heading else translated)
    return "\n".join(output)


def write_reports(destination, text):
    destination = Path(destination)
    chinese = destination.with_name(destination.stem + ".zh-CN" + destination.suffix)
    destination.write_text(text, encoding="utf-8")
    chinese.write_text(chinese_markdown(text), encoding="utf-8")
