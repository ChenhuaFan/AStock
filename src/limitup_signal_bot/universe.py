from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass(frozen=True)
class Stock:
    ticker: str
    name: str


def normalize_ticker(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(re.findall(r"\d+", text))
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _cell_col(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def _cell_row(ref: str) -> int:
    text = "".join(ch for ch in ref if ch.isdigit())
    return int(text) if text else 0


def _read_cell(c: ET.Element, shared_strings: list[str]) -> str:
    value = c.find("a:v", NS)
    inline = c.find("a:is", NS)
    if c.attrib.get("t") == "s" and value is not None:
        return shared_strings[int(value.text or "0")]
    if c.attrib.get("t") == "inlineStr" and inline is not None:
        return "".join(
            (t.text or "")
            for t in inline.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )
    return value.text if value is not None and value.text is not None else ""


def load_universe(xlsx_path: Path) -> list[Stock]:
    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared_strings.append(
                    "".join(
                        (t.text or "")
                        for t in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    )
                )

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = workbook.find("a:sheets", NS)[0]
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rid_to_target[rid]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet = ET.fromstring(z.read(sheet_path))

    stocks: list[Stock] = []
    seen = set()
    for row in sheet.findall(".//a:sheetData/a:row", NS):
        row_index = _cell_row(row.attrib.get("r", "0"))
        if row_index < 2:
            continue
        values = {}
        for cell in row.findall("a:c", NS):
            values[_cell_col(cell.attrib.get("r", ""))] = _read_cell(cell, shared_strings)
        code = normalize_ticker(values.get("B", ""))
        if len(code) != 6 or code in seen:
            continue
        name = str(values.get("C", "")).strip()
        seen.add(code)
        stocks.append(Stock(ticker=code, name=name))
    return stocks
