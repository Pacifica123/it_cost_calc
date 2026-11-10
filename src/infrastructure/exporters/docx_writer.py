"""Minimal DOCX (Office Open XML) writer on the standard library.

Supports what a report needs: headings, paragraphs (with **bold** fragments), bullet and
numbered lines, monospace blocks and tables. The file opens in Word, LibreOffice and
«Мой Офис»; fonts are Times New Roman 14 / 12 for tables, as in a thesis appendix.
"""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/><w:sz w:val="28"/><w:szCs w:val="28"/><w:lang w:val="ru-RU"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="100"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="80"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:i/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="ReportTable"><w:name w:val="Report Table"/><w:tblPr><w:tblBorders>
<w:top w:val="single" w:sz="4" w:color="808080"/><w:left w:val="single" w:sz="4" w:color="808080"/><w:bottom w:val="single" w:sz="4" w:color="808080"/>
<w:right w:val="single" w:sz="4" w:color="808080"/><w:insideH w:val="single" w:sz="4" w:color="808080"/><w:insideV w:val="single" w:sz="4" w:color="808080"/>
</w:tblBorders><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>"""

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_PAGE_WIDTH_TWIPS = 9638  # A4 minus 30 mm + 15 mm margins


def _runs(text: str, *, bold: bool = False) -> str:
    parts: list[str] = []
    position = 0
    for match in _BOLD.finditer(text):
        if match.start() > position:
            parts.append(_run(text[position : match.start()], bold))
        parts.append(_run(match.group(1), True))
        position = match.end()
    if position < len(text) or not parts:
        parts.append(_run(text[position:], bold))
    return "".join(parts)


def _run(text: str, bold: bool) -> str:
    properties = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:r>{properties}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


class DocxDocument:
    """Collects blocks and writes them as a .docx file."""

    def __init__(self, title: str = "Отчёт") -> None:
        self.title = title
        self._body: list[str] = []

    # --- blocks ---------------------------------------------------------------------------------

    def heading(self, text: str, level: int = 1) -> None:
        style = "Title" if level <= 0 else f"Heading{min(level, 3)}"
        self._body.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{_runs(text)}</w:p>')

    def paragraph(self, text: str, *, bold: bool = False, indent: bool = True) -> None:
        first_line = '<w:ind w:firstLine="709"/>' if indent else ""
        self._body.append(f'<w:p><w:pPr><w:jc w:val="both"/>{first_line}</w:pPr>{_runs(text, bold=bold)}</w:p>')

    def bullet(self, text: str, marker: str = "–") -> None:
        self._body.append(
            f'<w:p><w:pPr><w:ind w:left="709" w:hanging="284"/></w:pPr>{_run(marker + " ", False)}{_runs(text)}</w:p>'
        )

    def code(self, lines: Sequence[str]) -> None:
        for line in lines:
            self._body.append(f'<w:p><w:pPr><w:pStyle w:val="Code"/></w:pPr>{_run(line, False)}</w:p>')

    def table(self, header: Sequence[str], rows: Sequence[Sequence[str]], *, caption: str | None = None) -> None:
        if caption:
            self._body.append(f'<w:p><w:pPr><w:keepNext/></w:pPr>{_run(caption, False)}</w:p>')
        columns = max([len(header)] + [len(row) for row in rows]) if (header or rows) else 0
        if columns == 0:
            return
        width = _PAGE_WIDTH_TWIPS // columns
        grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(columns))
        xml = [
            '<w:tbl><w:tblPr><w:tblStyle w:val="ReportTable"/><w:tblW w:w="5000" w:type="pct"/></w:tblPr>',
            f"<w:tblGrid>{grid}</w:tblGrid>",
        ]
        if header:
            xml.append(self._row(header, columns, width, bold=True, repeat=True))
        for row in rows:
            xml.append(self._row(row, columns, width))
        xml.append("</w:tbl>")
        self._body.append("".join(xml))
        self._body.append("<w:p/>")

    @staticmethod
    def _row(cells: Sequence[str], columns: int, width: int, *, bold: bool = False, repeat: bool = False) -> str:
        properties = "<w:trPr><w:tblHeader/></w:trPr>" if repeat else ""
        values = list(cells) + [""] * (columns - len(cells))
        out = [f"<w:tr>{properties}"]
        for value in values:
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>{_runs(str(value), bold=bold)}</w:p></w:tc>'
            )
        out.append("</w:tr>")
        return "".join(out)

    def page_break(self) -> None:
        self._body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    # --- markdown -------------------------------------------------------------------------------

    def add_markdown(self, markdown: str) -> None:
        """Converts the subset of Markdown used by the project's reports."""
        lines = markdown.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].rstrip()
            stripped = line.strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("```"):
                block: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    block.append(lines[index].rstrip())
                    index += 1
                self.code(block[:200])
                index += 1
                continue
            if stripped.startswith("|"):
                table_lines: list[str] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index].strip())
                    index += 1
                cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
                cells = [row for row in cells if not all(re.fullmatch(r":?-{2,}:?", cell or "--") for cell in row)]
                if cells:
                    self.table(cells[0], cells[1:])
                continue
            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading:
                level = len(heading.group(1))
                # "#" is the document title, "##" chapters, "###" sections.
                if level == 1:
                    self.heading(heading.group(2), 0 if not self._body else 1)
                else:
                    self.heading(heading.group(2), level - 1)
                index += 1
                continue
            bullet = re.match(r"^[-*]\s+(.*)$", stripped)
            if bullet:
                self.bullet(bullet.group(1))
                index += 1
                continue
            numbered = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
            if numbered:
                self.bullet(numbered.group(2), marker=f"{numbered.group(1)}.")
                index += 1
                continue
            self.paragraph(stripped.replace("`", ""))
            index += 1

    # --- output ---------------------------------------------------------------------------------

    def document_xml(self) -> str:
        section = (
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="1701" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{_W}"><w:body>{"".join(self._body)}{section}</w:body></w:document>'
        )

    def save(self, filename: str | Path) -> Path:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        core = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{escape(self.title)}</dc:title><dc:creator>IT Cost Calc</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'
        )
        app = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>IT Cost Calc</Application></Properties>"
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
            archive.writestr("_rels/.rels", _ROOT_RELS)
            archive.writestr("word/_rels/document.xml.rels", _DOC_RELS)
            archive.writestr("word/styles.xml", _STYLES)
            archive.writestr("word/document.xml", self.document_xml())
            archive.writestr("docProps/core.xml", core)
            archive.writestr("docProps/app.xml", app)
        return path
