"""Minimal Excel workbook writer for tabular benchmark output."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

ChartSeries = tuple[Any, ...]
ChartSpec = tuple[Any, ...]
FormulaCell = tuple[str, str, Any]
Sheet = tuple[str, list[str], list[list[Any]]] | tuple[str, list[str], list[list[Any]], list[ChartSpec]]


def write_xlsx(path: Path, headers: list[str], rows: list[list[Any]], sheet_name: str = "Table 1") -> None:
    """Write a single-sheet .xlsx file without external dependencies."""
    write_workbook(path, [(sheet_name, headers, rows)])


def write_workbook(path: Path, sheets: list[Sheet]) -> None:
    """Write a multi-sheet .xlsx workbook without external dependencies."""
    if not sheets:
        raise ValueError("At least one sheet is required.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        chart_count = sum(len(_sheet_charts(sheet)) for sheet in sheets)
        drawing_count = sum(1 for sheet in sheets if _sheet_charts(sheet))
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets), chart_count, drawing_count))
        archive.writestr("_rels/.rels", _root_relationships_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml([sheet[0] for sheet in sheets]))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships_xml(len(sheets)))
        archive.writestr("xl/styles.xml", _styles_xml())
        next_chart_index = 1
        next_drawing_index = 1
        for index, sheet in enumerate(sheets, start=1):
            _, headers, rows = sheet[:3]
            charts = _sheet_charts(sheet)
            drawing_index = next_drawing_index if charts else None
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(headers, rows, drawing_index))
            if charts:
                archive.writestr(
                    f"xl/worksheets/_rels/sheet{index}.xml.rels",
                    _worksheet_relationships_xml(drawing_index),
                )
                chart_indexes = list(range(next_chart_index, next_chart_index + len(charts)))
                archive.writestr(
                    f"xl/drawings/drawing{drawing_index}.xml",
                    _drawing_xml(charts, chart_indexes),
                )
                archive.writestr(
                    f"xl/drawings/_rels/drawing{drawing_index}.xml.rels",
                    _drawing_relationships_xml(chart_indexes),
                )
                for chart_index, chart in zip(chart_indexes, charts):
                    archive.writestr(f"xl/charts/chart{chart_index}.xml", _chart_xml(chart))
                next_chart_index += len(charts)
                next_drawing_index += 1


def _sheet_charts(sheet: Sheet) -> list[ChartSpec]:
    return sheet[3] if len(sheet) == 4 else []


def _content_types_xml(sheet_count: int, chart_count: int, drawing_count: int) -> str:
    worksheets = "\n".join(
        f'  <Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    charts = "\n".join(
        f'  <Override PartName="/xl/charts/chart{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        for i in range(1, chart_count + 1)
    )
    drawings = "\n".join(
        f'  <Override PartName="/xl/drawings/drawing{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
        for i in range(1, drawing_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{worksheets}
{drawings}
{charts}
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""


def _root_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "\n".join(
        f'    <sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet_name in enumerate(sheet_names, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
{sheets}
  </sheets>
</workbook>"""


def _workbook_relationships_xml(sheet_count: int) -> str:
    worksheets = "\n".join(
        f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    styles_id = sheet_count + 1
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{worksheets}
  <Relationship Id="rId{styles_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _worksheet_xml(headers: list[str], rows: list[list[Any]], drawing_index: int | None = None) -> str:
    column_widths = "".join(
        f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
        for i, width in enumerate([10, 10, 10, 12, 10, 12, 12, 10, 12, 10], start=1)
    )
    if headers:
        body_rows = [_row_xml(1, headers, style=1)]
        body_rows.extend(_row_xml(i, row) for i, row in enumerate(rows, start=2))
    else:
        body_rows = [_row_xml(i, row) for i, row in enumerate(rows, start=1)]
    drawing = f'<drawing r:id="rId1"/>' if drawing_index is not None else ""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <cols>{column_widths}</cols>
  <sheetData>{''.join(body_rows)}</sheetData>
  {drawing}
</worksheet>"""


def _row_xml(row_number: int, values: list[Any], style: int | None = None) -> str:
    cells = "".join(_cell_xml(row_number, column_index, value, style) for column_index, value in enumerate(values, start=1))
    return f'<row r="{row_number}">{cells}</row>'


def _cell_xml(row_number: int, column_index: int, value: Any, style: int | None = None) -> str:
    reference = f"{_column_name(column_index)}{row_number}"
    style_attr = f' s="{style}"' if style is not None else ""
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "formula":
        formula = escape(value[1])
        cached_value = value[2]
        if isinstance(cached_value, str):
            return f'<c r="{reference}" t="str"{style_attr}><f>{formula}</f><v>{escape(cached_value)}</v></c>'
        return f'<c r="{reference}"{style_attr}><f>{formula}</f><v>{cached_value}</v></c>'
    if isinstance(value, str):
        return f'<c r="{reference}" t="inlineStr"{style_attr}><is><t>{escape(value)}</t></is></c>'
    return f'<c r="{reference}"{style_attr}><v>{value}</v></c>'


def _column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _worksheet_relationships_xml(drawing_index: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing{drawing_index}.xml"/>
</Relationships>"""


def _drawing_relationships_xml(chart_indexes: list[int]) -> str:
    relationships = "\n".join(
        f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart{chart_index}.xml"/>'
        for i, chart_index in enumerate(chart_indexes, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{relationships}
</Relationships>"""


def _drawing_xml(charts: list[ChartSpec], chart_indexes: list[int]) -> str:
    anchors = []
    for relationship_index, chart in enumerate(charts, start=1):
        row = chart[1]
        column = chart[2]
        to_column = _chart_option(chart, "to_column", column + 10)
        to_row = _chart_option(chart, "to_row", row + 18)
        to_column_offset = _chart_option(chart, "to_column_offset", 0)
        to_row_offset = _chart_option(chart, "to_row_offset", 0)
        drawing_id = _chart_option(chart, "drawing_id", relationship_index)
        drawing_name = _chart_option(chart, "drawing_name", f"Chart {relationship_index}")
        drawing_ext_xml = _chart_option(chart, "drawing_ext_xml", "")
        anchors.append(f"""
  <xdr:twoCellAnchor>
    <xdr:from><xdr:col>{column - 1}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{row - 1}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>
    <xdr:to><xdr:col>{to_column}</xdr:col><xdr:colOff>{to_column_offset}</xdr:colOff><xdr:row>{to_row}</xdr:row><xdr:rowOff>{to_row_offset}</xdr:rowOff></xdr:to>
    <xdr:graphicFrame macro="">
      <xdr:nvGraphicFramePr>
        <xdr:cNvPr id="{drawing_id}" name="{escape(drawing_name)}">{drawing_ext_xml}</xdr:cNvPr>
        <xdr:cNvGraphicFramePr/>
      </xdr:nvGraphicFramePr>
      <xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>
      <a:graphic>
        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">
          <c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId{relationship_index}"/>
        </a:graphicData>
      </a:graphic>
    </xdr:graphicFrame>
    <xdr:clientData/>
  </xdr:twoCellAnchor>""")
        for text_box_index, text_box in enumerate(_chart_option(chart, "text_boxes", ()), start=1):
            anchors.append(_text_box_xml(text_box, drawing_id + 1000 + text_box_index))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
{''.join(anchors)}
</xdr:wsDr>"""


def _text_box_xml(text_box: dict[str, Any], drawing_id: int) -> str:
    row = text_box["row"]
    column = text_box["column"]
    to_row = text_box["to_row"]
    to_column = text_box["to_column"]
    name = text_box.get("name", f"Text Box {drawing_id}")
    fill_color = text_box.get("fill_color", "FFFFFF")
    line_color = text_box.get("line_color", "808080")
    paragraphs = "".join(
        f'<a:p><a:r><a:rPr sz="900"/><a:t>{escape(line)}</a:t></a:r></a:p>'
        for line in str(text_box["text"]).splitlines()
    )
    return f"""
  <xdr:twoCellAnchor>
    <xdr:from><xdr:col>{column - 1}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{row - 1}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>
    <xdr:to><xdr:col>{to_column - 1}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{to_row - 1}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>
    <xdr:sp macro="" textlink="">
      <xdr:nvSpPr><xdr:cNvPr id="{drawing_id}" name="{escape(name)}"/><xdr:cNvSpPr txBox="1"/></xdr:nvSpPr>
      <xdr:spPr><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill_color}"/></a:solidFill><a:ln><a:solidFill><a:srgbClr val="{line_color}"/></a:solidFill></a:ln></xdr:spPr>
      <xdr:txBody><a:bodyPr wrap="square" rtlCol="0"/><a:lstStyle/>{paragraphs}</xdr:txBody>
    </xdr:sp>
    <xdr:clientData/>
  </xdr:twoCellAnchor>"""


def _chart_options(chart: ChartSpec) -> dict[str, Any]:
    return chart[-1] if chart and isinstance(chart[-1], dict) else {}


def _chart_option(chart: ChartSpec, name: str, default: Any) -> Any:
    return _chart_options(chart).get(name, default)


def _chart_xml(chart: ChartSpec) -> str:
    chart_type = chart[4] if len(chart) >= 5 else "line"
    if chart_type == "scatter":
        return _scatter_chart_xml(chart)
    if chart_type == "column":
        return _column_chart_xml(chart)
    if chart_type == "line_high_low":
        return _line_high_low_chart_xml(chart)
    if chart_type == "line_secondary":
        return _line_secondary_chart_xml(chart)

    title = chart[0]
    series = chart[3]
    y_axis_scaling = _axis_scaling_xml(chart)
    x_axis_title = _axis_title_xml(_x_axis_title(chart), "x")
    y_axis_title = _axis_title_xml(_y_axis_title(chart), "y")
    x_axis_num_format = _axis_num_format_xml(_x_axis_num_format(chart, chart_type))
    y_axis_num_format = _axis_num_format_xml(_y_axis_num_format(chart, chart_type))
    series_xml = "\n".join(_chart_series_xml(index, item) for index, item in enumerate(series))
    legend = _legend_xml(chart, chart_type)
    x_axis = _line_x_axis_xml(chart, x_axis_title, x_axis_num_format)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <c:style val="2"/>
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:layout/>
      <c:lineChart>
        <c:grouping val="standard"/>
{series_xml}
        <c:axId val="123456"/>
        <c:axId val="654321"/>
      </c:lineChart>
      {x_axis}
      <c:valAx><c:axId val="654321"/>{y_axis_scaling}<c:delete val="0"/><c:axPos val="l"/>{y_axis_title}{y_axis_num_format}<c:crossAx val="123456"/><c:tickLblPos val="nextTo"/></c:valAx>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
  </c:chart>
</c:chartSpace>"""


def _axis_scaling_xml(chart: ChartSpec) -> str:
    if len(chart) >= 7 and chart[5] is not None and chart[6] is not None:
        return f'<c:scaling><c:orientation val="minMax"/><c:min val="{chart[5]}"/><c:max val="{chart[6]}"/></c:scaling>'
    return '<c:scaling><c:orientation val="minMax"/></c:scaling>'


def _x_axis_scaling_xml(chart: ChartSpec) -> str:
    if len(chart) >= 11 and chart[9] is not None and chart[10] is not None:
        return f'<c:scaling><c:orientation val="minMax"/><c:min val="{chart[9]}"/><c:max val="{chart[10]}"/></c:scaling>'
    return '<c:scaling><c:orientation val="minMax"/></c:scaling>'


def _x_axis_title(chart: ChartSpec) -> str:
    return chart[7] if len(chart) >= 9 else ""


def _y_axis_title(chart: ChartSpec) -> str:
    return chart[8] if len(chart) >= 9 else ""


def _x_axis_num_format(chart: ChartSpec, chart_type: str) -> str:
    if chart_type == "scatter":
        return chart[11] if len(chart) >= 13 else "0.00"
    return chart[9] if len(chart) >= 11 else "0.00"


def _y_axis_num_format(chart: ChartSpec, chart_type: str) -> str:
    if chart_type == "scatter":
        return chart[12] if len(chart) >= 13 else "0.00"
    return chart[10] if len(chart) >= 11 else "0.00"


def _show_legend(chart: ChartSpec, chart_type: str) -> bool:
    if chart_type == "scatter":
        return chart[13] if len(chart) >= 14 else True
    return chart[11] if len(chart) >= 12 else True


def _line_x_axis_xml(chart: ChartSpec, x_axis_title: str, x_axis_num_format: str) -> str:
    text_rotation = _axis_text_rotation_xml(chart)
    if len(chart) >= 14 and chart[13] == "date_months":
        major_unit = chart[12] if len(chart) >= 13 and chart[12] else 3
        return (
            '<c:dateAx><c:axId val="123456"/>'
            '<c:scaling><c:orientation val="minMax"/></c:scaling>'
            '<c:delete val="0"/><c:axPos val="b"/>'
            f'{x_axis_title}{x_axis_num_format}'
            '<c:crossAx val="654321"/><c:tickLblPos val="nextTo"/>'
            '<c:baseTimeUnit val="months"/>'
            f'<c:majorUnit val="{major_unit}"/><c:majorTimeUnit val="months"/>'
            f'{text_rotation}'
            '</c:dateAx>'
        )
    tick_skip = ""
    if len(chart) >= 13 and chart[12]:
        tick_skip = f'<c:tickLblSkip val="{chart[12]}"/>'
    return (
        '<c:catAx><c:axId val="123456"/>'
        '<c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="b"/>'
        f'{x_axis_title}{x_axis_num_format}'
        f'<c:crossAx val="654321"/><c:tickLblPos val="nextTo"/>{tick_skip}{text_rotation}</c:catAx>'
    )


def _x_axis_tick_skip_xml(chart: ChartSpec, chart_type: str) -> str:
    if chart_type == "scatter":
        return ""
    if len(chart) >= 13 and chart[12]:
        return f'<c:tickLblSkip val="{chart[12]}"/>'
    return ""


def _axis_num_format_xml(format_code: str) -> str:
    return f'<c:numFmt formatCode="{escape(format_code)}" sourceLinked="0"/>'


def _axis_text_rotation_xml(chart: ChartSpec) -> str:
    if len(chart) < 15 or chart[14] is None:
        return ""
    rotation = int(float(chart[14]) * -60000)
    return (
        f'<c:txPr><a:bodyPr rot="{rotation}"/><a:lstStyle/>'
        '<a:p><a:pPr><a:defRPr/></a:pPr><a:endParaRPr/></a:p></c:txPr>'
    )


def _legend_xml(chart: ChartSpec, chart_type: str) -> str:
    if not _show_legend(chart, chart_type):
        return ""
    hidden_count = chart[13] if chart_type != "scatter" and len(chart) >= 14 and isinstance(chart[13], int) else 0
    hidden_entries = "".join(
        f'<c:legendEntry><c:idx val="{index}"/><c:delete val="1"/></c:legendEntry>'
        for index in range(hidden_count)
    )
    layout = _chart_option(chart, "legend_layout_xml", "<c:layout/>")
    overlay = ""
    if "legend_overlay" in _chart_options(chart):
        overlay = f'<c:overlay val="{int(bool(_chart_option(chart, "legend_overlay", False)))}"/>'
    return f'<c:legend><c:legendPos val="r"/>{hidden_entries}{layout}{overlay}</c:legend>'


def _axis_title_xml(title: str, axis: str) -> str:
    if not title:
        return ""
    return f'<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>'


def _scatter_chart_xml(chart: ChartSpec) -> str:
    if _chart_option(chart, "excel_saved_chart_style", "") == "p3_mc_runs":
        return _excel_saved_p3_scatter_chart_xml(chart)

    title = chart[0]
    series = chart[3]
    scatter_style = _chart_option(chart, "scatter_style", "marker")
    x_axis_scaling = _x_axis_scaling_xml(chart)
    y_axis_scaling = _axis_scaling_xml(chart)
    x_axis_title = _axis_title_xml(_x_axis_title(chart), "x")
    y_axis_title = _axis_title_xml(_y_axis_title(chart), "y")
    x_axis_num_format = _axis_num_format_xml(_x_axis_num_format(chart, "scatter"))
    y_axis_num_format = _axis_num_format_xml(_y_axis_num_format(chart, "scatter"))
    series_xml = "\n".join(_scatter_series_xml(index, item) for index, item in enumerate(series))
    legend = _legend_xml(chart, "scatter")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <c:style val="2"/>
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:layout/>
      <c:scatterChart>
        <c:scatterStyle val="{scatter_style}"/>
{series_xml}
        <c:axId val="123456"/>
        <c:axId val="654321"/>
      </c:scatterChart>
      <c:valAx><c:axId val="123456"/>{x_axis_scaling}<c:delete val="0"/><c:axPos val="b"/>{x_axis_title}{x_axis_num_format}<c:crossAx val="654321"/><c:tickLblPos val="nextTo"/></c:valAx>
      <c:valAx><c:axId val="654321"/>{y_axis_scaling}<c:delete val="0"/><c:axPos val="l"/>{y_axis_title}{y_axis_num_format}<c:crossAx val="123456"/><c:tickLblPos val="nextTo"/></c:valAx>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
  </c:chart>
</c:chartSpace>"""


def _excel_saved_p3_scatter_chart_xml(chart: ChartSpec) -> str:
    title = chart[0]
    series = chart[3]
    scatter_style = _chart_option(chart, "scatter_style", "lineMarker")
    x_axis_id = _chart_option(chart, "x_axis_id", 1510823840)
    y_axis_id = _chart_option(chart, "y_axis_id", 156514608)
    x_axis_min = _format_chart_number(chart[9]) if len(chart) >= 11 and chart[9] is not None else None
    x_axis_max = _format_chart_number(chart[10]) if len(chart) >= 11 and chart[10] is not None else None
    y_axis_min = _format_chart_number(chart[5]) if len(chart) >= 7 and chart[5] is not None else None
    y_axis_max = _format_chart_number(chart[6]) if len(chart) >= 7 and chart[6] is not None else None
    x_axis_scaling = _saved_axis_scaling_xml(x_axis_min, x_axis_max)
    y_axis_scaling = _saved_axis_scaling_xml(y_axis_min, y_axis_max)
    x_axis_title = _x_axis_title(chart)
    y_axis_title = _y_axis_title(chart)
    x_axis_num_format = _x_axis_num_format(chart, "scatter")
    y_axis_num_format = _y_axis_num_format(chart, "scatter")
    vertical_lines = _chart_option(chart, "vertical_lines", ())
    series_xml = "\n".join(_saved_excel_scatter_series_xml(index, item) for index, item in enumerate(series))
    vertical_line_xml = "\n".join(
        _saved_vertical_line_series_xml(index + len(series) + 1, line)
        for index, line in enumerate(vertical_lines)
    )
    hidden_legend_indexes = tuple(range(len(series), len(series) + len(vertical_lines)))
    legend = _saved_excel_legend_xml(hidden_legend_indexes) if _show_legend(chart, "scatter") else ""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:c16r2="http://schemas.microsoft.com/office/drawing/2015/06/chart">
  <c:date1904 val="0"/>
  <c:lang val="en-US"/>
  <c:roundedCorners val="0"/>
  <mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">
    <mc:Choice xmlns:c14="http://schemas.microsoft.com/office/drawing/2007/8/2/chart" Requires="c14">
      <c14:style val="102"/>
    </mc:Choice>
    <mc:Fallback>
      <c:style val="2"/>
    </mc:Fallback>
  </mc:AlternateContent>
  <c:chart>
    {_saved_excel_chart_title_xml(title)}
    <c:autoTitleDeleted val="0"/>
    <c:plotArea>
      <c:layout/>
      <c:scatterChart>
        <c:scatterStyle val="{scatter_style}"/>
        <c:varyColors val="0"/>
{series_xml}
{vertical_line_xml}
        <c:dLbls><c:showLegendKey val="0"/><c:showVal val="0"/><c:showCatName val="0"/><c:showSerName val="0"/><c:showPercent val="0"/><c:showBubbleSize val="0"/></c:dLbls>
        <c:axId val="{x_axis_id}"/>
        <c:axId val="{y_axis_id}"/>
      </c:scatterChart>
      <c:valAx>
        <c:axId val="{x_axis_id}"/>
        {x_axis_scaling}
        <c:delete val="0"/>
        <c:axPos val="b"/>
        {_saved_major_gridlines_xml()}
        {_saved_excel_axis_title_xml(x_axis_title, _saved_body_pr_attrs())}
        <c:numFmt formatCode="{escape(x_axis_num_format)}" sourceLinked="1"/>
        <c:majorTickMark val="none"/>
        <c:minorTickMark val="none"/>
        <c:tickLblPos val="nextTo"/>
        {_saved_axis_line_xml()}
        {_saved_axis_tick_text_xml('-2700000', include_vert=False)}
        <c:crossAx val="{y_axis_id}"/>
        <c:crosses val="autoZero"/>
        <c:crossBetween val="midCat"/>
        <c:majorUnit val="1"/>
      </c:valAx>
      <c:valAx>
        <c:axId val="{y_axis_id}"/>
        {y_axis_scaling}
        <c:delete val="0"/>
        <c:axPos val="l"/>
        {_saved_major_gridlines_xml()}
        {_saved_excel_axis_title_xml(y_axis_title, _saved_body_pr_attrs(rot='-5400000'))}
        <c:numFmt formatCode="{escape(y_axis_num_format)}" sourceLinked="1"/>
        <c:majorTickMark val="none"/>
        <c:minorTickMark val="none"/>
        <c:tickLblPos val="nextTo"/>
        {_saved_axis_line_xml()}
        {_saved_axis_tick_text_xml('-60000000')}
        <c:crossAx val="{x_axis_id}"/>
        <c:crosses val="autoZero"/>
        <c:crossBetween val="midCat"/>
      </c:valAx>
      <c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln><a:noFill/></a:ln><a:effectLst/></c:spPr>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
    <c:dispBlanksAs val="gap"/>
    <c:showDLblsOverMax val="0"/>
    <c:extLst><c:ext xmlns:c16r3="http://schemas.microsoft.com/office/drawing/2017/03/chart" uri="{{56B9EC1D-385E-4148-901F-78D8002777C0}}"><c16r3:dataDisplayOptions16><c16r3:dispNaAsBlank val="1"/></c16r3:dataDisplayOptions16></c:ext></c:extLst>
  </c:chart>
  <c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="tx1"><a:lumMod val="15000"/><a:lumOff val="85000"/></a:schemeClr></a:solidFill><a:round/></a:ln><a:effectLst/></c:spPr>
  <c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr/></a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr>
  <c:printSettings><c:headerFooter/><c:pageMargins b="0.75" l="0.7" r="0.7" t="0.75" header="0.3" footer="0.3"/><c:pageSetup/></c:printSettings>
</c:chartSpace>"""


def _format_chart_number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _saved_axis_scaling_xml(min_value: str | None, max_value: str | None) -> str:
    bounds = ""
    if max_value is not None:
        bounds += f'<c:max val="{max_value}"/>'
    if min_value is not None:
        bounds += f'<c:min val="{min_value}"/>'
    return f"<c:scaling><c:orientation val=\"minMax\"/>{bounds}</c:scaling>"


def _saved_body_pr_attrs(rot: str = "0") -> str:
    return f'rot="{rot}" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" anchorCtr="1"'


def _saved_def_rpr_xml(size: int, *, include_spc: bool = False) -> str:
    spc = ' spc="0"' if include_spc else ""
    return (
        f'<a:defRPr sz="{size}" b="0" i="0" u="none" strike="noStrike" kern="1200"{spc} baseline="0">'
        '<a:solidFill><a:schemeClr val="tx1"><a:lumMod val="65000"/><a:lumOff val="35000"/></a:schemeClr></a:solidFill>'
        '<a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/><a:cs typeface="+mn-cs"/></a:defRPr>'
    )


def _saved_rich_text_xml(text: str, size: int, body_pr_attrs: str, *, run_lang: str, include_spc: bool = False) -> str:
    return (
        '<c:tx><c:rich>'
        f'<a:bodyPr {body_pr_attrs}/><a:lstStyle/><a:p><a:pPr>{_saved_def_rpr_xml(size, include_spc=include_spc)}</a:pPr>'
        f'<a:r><a:rPr lang="{run_lang}"/><a:t>{escape(text)}</a:t></a:r></a:p>'
        '</c:rich></c:tx>'
    )


def _saved_tx_pr_xml(size: int, body_pr_attrs: str, *, include_spc: bool = False) -> str:
    return (
        '<c:txPr>'
        f'<a:bodyPr {body_pr_attrs}/><a:lstStyle/><a:p><a:pPr>{_saved_def_rpr_xml(size, include_spc=include_spc)}</a:pPr>'
        '<a:endParaRPr lang="en-US"/></a:p>'
        '</c:txPr>'
    )


def _saved_title_shape_xml() -> str:
    return '<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln><a:effectLst/></c:spPr>'


def _saved_excel_chart_title_xml(title: str) -> str:
    body_pr = _saved_body_pr_attrs()
    return (
        '<c:title>'
        f'{_saved_rich_text_xml(title, 1400, body_pr, run_lang="en-US", include_spc=True)}'
        '<c:overlay val="0"/>'
        f'{_saved_title_shape_xml()}'
        f'{_saved_tx_pr_xml(1400, body_pr, include_spc=True)}'
        '</c:title>'
    )


def _saved_excel_axis_title_xml(title: str, body_pr_attrs: str) -> str:
    if not title:
        return ""
    return (
        '<c:title>'
        f'{_saved_rich_text_xml(title, 1000, body_pr_attrs, run_lang="en-GB")}'
        '<c:overlay val="0"/>'
        f'{_saved_title_shape_xml()}'
        f'{_saved_tx_pr_xml(1000, body_pr_attrs)}'
        '</c:title>'
    )


def _saved_major_gridlines_xml() -> str:
    return (
        '<c:majorGridlines><c:spPr><a:ln w="9525" cap="flat" cmpd="sng" algn="ctr">'
        '<a:solidFill><a:schemeClr val="tx1"><a:lumMod val="15000"/><a:lumOff val="85000"/></a:schemeClr></a:solidFill>'
        '<a:round/></a:ln><a:effectLst/></c:spPr></c:majorGridlines>'
    )


def _saved_axis_line_xml() -> str:
    return (
        '<c:spPr><a:noFill/><a:ln w="9525" cap="flat" cmpd="sng" algn="ctr">'
        '<a:solidFill><a:schemeClr val="tx1"><a:lumMod val="25000"/><a:lumOff val="75000"/></a:schemeClr></a:solidFill>'
        '<a:round/></a:ln><a:effectLst/></c:spPr>'
    )


def _saved_axis_tick_text_xml(rotation: str, *, include_vert: bool = True) -> str:
    vert = ' vert="horz"' if include_vert else ""
    return (
        '<c:txPr>'
        f'<a:bodyPr rot="{rotation}" spcFirstLastPara="1" vertOverflow="ellipsis"{vert} wrap="square" anchor="ctr" anchorCtr="1"/>'
        f'<a:lstStyle/><a:p><a:pPr>{_saved_def_rpr_xml(900)}</a:pPr><a:endParaRPr lang="en-US"/></a:p>'
        '</c:txPr>'
    )


def _saved_excel_legend_xml(hidden_entry_indexes: tuple[int, ...] = ()) -> str:
    body_pr = _saved_body_pr_attrs()
    hidden_entries = "".join(
        f'<c:legendEntry><c:idx val="{index}"/><c:delete val="1"/></c:legendEntry>'
        for index in hidden_entry_indexes
    )
    return (
        f'<c:legend><c:legendPos val="r"/>{hidden_entries}<c:overlay val="0"/>'
        f'{_saved_title_shape_xml()}'
        f'{_saved_tx_pr_xml(900, body_pr)}'
        '</c:legend>'
    )


def _saved_excel_scatter_series_xml(index: int, series: ChartSeries) -> str:
    name, x_ref, y_ref, error_ref = series
    marker_styles = (
        ("circle", "accent2"),
        ("diamond", "accent3"),
        ("square", "accent4"),
        ("triangle", "accent5"),
        ("x", "accent6"),
        ("plus", "accent1"),
    )
    marker_symbol, marker_color = marker_styles[index % len(marker_styles)]
    error_bars = _saved_error_bars_xml(error_ref, marker_color)
    return f"""
        <c:ser>
          <c:idx val="{index + 1}"/>
          <c:order val="{index}"/>
          <c:tx><c:v>{escape(name)}</c:v></c:tx>
          <c:spPr><a:ln w="25400" cap="rnd"><a:noFill/><a:round/></a:ln><a:effectLst/></c:spPr>
          <c:marker><c:symbol val="{marker_symbol}"/><c:size val="8"/><c:spPr><a:solidFill><a:schemeClr val="{marker_color}"/></a:solidFill><a:ln w="9525"><a:solidFill><a:schemeClr val="{marker_color}"/></a:solidFill></a:ln><a:effectLst/></c:spPr></c:marker>
{error_bars}
          <c:xVal><c:numRef><c:f>{escape(x_ref)}</c:f></c:numRef></c:xVal>
          <c:yVal><c:numRef><c:f>{escape(y_ref)}</c:f></c:numRef></c:yVal>
          <c:smooth val="0"/>
        </c:ser>"""


def _saved_error_bars_xml(error_ref: str, marker_color: str) -> str:
    if not error_ref:
        return ""
    return f"""
          <c:errBars>
            <c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/>
            <c:noEndCap val="0"/>
            <c:plus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:plus>
            <c:minus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:minus>
            <c:spPr><a:ln w="9525"><a:solidFill><a:schemeClr val="{marker_color}"/></a:solidFill></a:ln></c:spPr>
          </c:errBars>"""


def _saved_vertical_line_series_xml(index: int, line: tuple[Any, ...]) -> str:
    x_value, y_min, y_max = line[:3]
    line_color = line[3] if len(line) >= 4 else "808080"
    line_width = line[4] if len(line) >= 5 else 12700
    x_value = _format_chart_number(x_value)
    y_min = _format_chart_number(y_min)
    y_max = _format_chart_number(y_max)
    return f"""
        <c:ser>
          <c:idx val="{index}"/>
          <c:order val="{index}"/>
          <c:tx><c:v>Vertical line {escape(x_value)}</c:v></c:tx>
          <c:spPr><a:ln w="{line_width}" cap="flat"><a:solidFill><a:srgbClr val="{line_color}"/></a:solidFill></a:ln></c:spPr>
          <c:marker><c:symbol val="none"/></c:marker>
          <c:xVal><c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="2"/><c:pt idx="0"><c:v>{escape(x_value)}</c:v></c:pt><c:pt idx="1"><c:v>{escape(x_value)}</c:v></c:pt></c:numLit></c:xVal>
          <c:yVal><c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="2"/><c:pt idx="0"><c:v>{escape(y_min)}</c:v></c:pt><c:pt idx="1"><c:v>{escape(y_max)}</c:v></c:pt></c:numLit></c:yVal>
          <c:smooth val="0"/>
        </c:ser>"""


def _line_high_low_chart_xml(chart: ChartSpec) -> str:
    title = chart[0]
    series = chart[3]
    x_axis_id = _chart_option(chart, "x_axis_id", 1510823840)
    y_axis_id = _chart_option(chart, "y_axis_id", 156514608)
    y_axis_min = _format_chart_number(chart[5]) if len(chart) >= 7 and chart[5] is not None else None
    y_axis_max = _format_chart_number(chart[6]) if len(chart) >= 7 and chart[6] is not None else None
    y_axis_scaling = _saved_axis_scaling_xml(y_axis_min, y_axis_max)
    x_axis_title = _x_axis_title(chart)
    y_axis_title = _y_axis_title(chart)
    x_axis_num_format = _axis_num_format_xml(_x_axis_num_format(chart, "line"))
    y_axis_num_format = _axis_num_format_xml(_y_axis_num_format(chart, "line"))
    series_xml = "\n".join(_high_low_line_series_xml(index, item) for index, item in enumerate(series))
    legend = _saved_excel_legend_xml() if _show_legend(chart, "line") else ""
    high_low_line_color = _chart_option(chart, "high_low_line_color", "808080")
    high_low_line_width = _chart_option(chart, "high_low_line_width", 12700)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:c16r2="http://schemas.microsoft.com/office/drawing/2015/06/chart">
  <c:date1904 val="0"/>
  <c:lang val="en-US"/>
  <c:roundedCorners val="0"/>
  <mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">
    <mc:Choice xmlns:c14="http://schemas.microsoft.com/office/drawing/2007/8/2/chart" Requires="c14">
      <c14:style val="102"/>
    </mc:Choice>
    <mc:Fallback>
      <c:style val="2"/>
    </mc:Fallback>
  </mc:AlternateContent>
  <c:chart>
    {_saved_excel_chart_title_xml(title)}
    <c:autoTitleDeleted val="0"/>
    <c:plotArea>
      <c:layout/>
      <c:lineChart>
        <c:grouping val="standard"/>
        <c:varyColors val="0"/>
{series_xml}
        <c:dLbls><c:showLegendKey val="0"/><c:showVal val="0"/><c:showCatName val="0"/><c:showSerName val="0"/><c:showPercent val="0"/><c:showBubbleSize val="0"/></c:dLbls>
        <c:hiLowLines><c:spPr><a:ln w="{high_low_line_width}" cap="flat"><a:solidFill><a:srgbClr val="{high_low_line_color}"/></a:solidFill></a:ln></c:spPr></c:hiLowLines>
        <c:axId val="{x_axis_id}"/>
        <c:axId val="{y_axis_id}"/>
      </c:lineChart>
      <c:catAx>
        <c:axId val="{x_axis_id}"/>
        <c:scaling><c:orientation val="minMax"/></c:scaling>
        <c:delete val="0"/>
        <c:axPos val="b"/>
        {_saved_excel_axis_title_xml(x_axis_title, _saved_body_pr_attrs())}
        {x_axis_num_format}
        <c:majorTickMark val="none"/>
        <c:minorTickMark val="none"/>
        <c:tickLblPos val="nextTo"/>
        {_saved_axis_line_xml()}
        {_saved_axis_tick_text_xml('-2700000', include_vert=False)}
        <c:crossAx val="{y_axis_id}"/>
        <c:crosses val="autoZero"/>
        <c:auto val="1"/>
        <c:lblAlgn val="ctr"/>
        <c:lblOffset val="100"/>
        <c:noMultiLvlLbl val="0"/>
      </c:catAx>
      <c:valAx>
        <c:axId val="{y_axis_id}"/>
        {y_axis_scaling}
        <c:delete val="0"/>
        <c:axPos val="l"/>
        {_saved_major_gridlines_xml()}
        {_saved_excel_axis_title_xml(y_axis_title, _saved_body_pr_attrs(rot='-5400000'))}
        {y_axis_num_format}
        <c:majorTickMark val="none"/>
        <c:minorTickMark val="none"/>
        <c:tickLblPos val="nextTo"/>
        {_saved_axis_line_xml()}
        {_saved_axis_tick_text_xml('-60000000')}
        <c:crossAx val="{x_axis_id}"/>
        <c:crosses val="autoZero"/>
        <c:crossBetween val="between"/>
      </c:valAx>
      <c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln><a:noFill/></a:ln><a:effectLst/></c:spPr>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
    <c:dispBlanksAs val="gap"/>
    <c:showDLblsOverMax val="0"/>
  </c:chart>
  <c:spPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="tx1"><a:lumMod val="15000"/><a:lumOff val="85000"/></a:schemeClr></a:solidFill><a:round/></a:ln><a:effectLst/></c:spPr>
  <c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr/></a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr>
  <c:printSettings><c:headerFooter/><c:pageMargins b="0.75" l="0.7" r="0.7" t="0.75" header="0.3" footer="0.3"/><c:pageSetup/></c:printSettings>
</c:chartSpace>"""


def _high_low_line_series_xml(index: int, series: ChartSeries) -> str:
    name, categories_ref, values_ref, _ = series
    marker_styles = (
        ("circle", "accent2"),
        ("diamond", "accent3"),
        ("square", "accent4"),
        ("triangle", "accent5"),
        ("x", "accent6"),
        ("plus", "accent1"),
    )
    marker_symbol, marker_color = marker_styles[index % len(marker_styles)]
    return f"""
        <c:ser>
          <c:idx val="{index}"/>
          <c:order val="{index}"/>
          <c:tx><c:v>{escape(name)}</c:v></c:tx>
          <c:spPr><a:ln w="25400" cap="rnd"><a:noFill/><a:round/></a:ln><a:effectLst/></c:spPr>
          <c:marker><c:symbol val="{marker_symbol}"/><c:size val="8"/><c:spPr><a:solidFill><a:schemeClr val="{marker_color}"/></a:solidFill><a:ln w="9525"><a:solidFill><a:schemeClr val="{marker_color}"/></a:solidFill></a:ln><a:effectLst/></c:spPr></c:marker>
          <c:cat><c:numRef><c:f>{escape(categories_ref)}</c:f></c:numRef></c:cat>
          <c:val><c:numRef><c:f>{escape(values_ref)}</c:f></c:numRef></c:val>
          <c:smooth val="0"/>
        </c:ser>"""


def _column_chart_xml(chart: ChartSpec) -> str:
    title = chart[0]
    series = chart[3]
    y_axis_scaling = _axis_scaling_xml(chart)
    x_axis_title = _axis_title_xml(_x_axis_title(chart), "x")
    y_axis_title = _axis_title_xml(_y_axis_title(chart), "y")
    x_axis_num_format = _axis_num_format_xml(_x_axis_num_format(chart, "column"))
    y_axis_num_format = _axis_num_format_xml(_y_axis_num_format(chart, "column"))
    series_xml = "\n".join(_column_series_xml(index, item) for index, item in enumerate(series))
    legend = _legend_xml(chart, "column")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <c:style val="2"/>
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:layout/>
      <c:barChart>
        <c:barDir val="col"/>
        <c:grouping val="clustered"/>
{series_xml}
        <c:axId val="123456"/>
        <c:axId val="654321"/>
      </c:barChart>
      {_line_x_axis_xml(chart, x_axis_title, x_axis_num_format)}
      <c:valAx><c:axId val="654321"/>{y_axis_scaling}<c:delete val="0"/><c:axPos val="l"/>{y_axis_title}{y_axis_num_format}<c:crossAx val="123456"/><c:tickLblPos val="nextTo"/></c:valAx>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
  </c:chart>
</c:chartSpace>"""


def _line_secondary_chart_xml(chart: ChartSpec) -> str:
    title = chart[0]
    primary_series = chart[3]
    secondary_series = chart[15]
    primary_series_xml = "\n".join(_chart_series_xml(index, item) for index, item in enumerate(primary_series))
    secondary_series_xml = "\n".join(
        _chart_series_xml(index + len(primary_series), item) for index, item in enumerate(secondary_series)
    )
    x_axis_title = _axis_title_xml(_x_axis_title(chart), "x")
    y_axis_title = _axis_title_xml(_y_axis_title(chart), "y")
    secondary_y_axis_title = _axis_title_xml(chart[16], "y")
    x_axis_num_format = _axis_num_format_xml(_x_axis_num_format(chart, "line"))
    y_axis_num_format = _axis_num_format_xml(_y_axis_num_format(chart, "line"))
    secondary_y_axis_num_format = _axis_num_format_xml(chart[17])
    legend = _legend_xml(chart, "line")
    primary_y_axis_format = _axis_num_format_xml(_chart_option(chart, "primary_y_axis_format", _y_axis_num_format(chart, "line")))
    secondary_y_axis_format = _axis_num_format_xml(_chart_option(chart, "secondary_y_axis_format", chart[17]))
    secondary_y_axis_id = _chart_option(chart, "secondary_y_axis_id", 1267093935)
    secondary_x_axis_id = _chart_option(chart, "secondary_x_axis_id", 1391646927)
    text_rotation = _axis_text_rotation_xml(chart)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <c:style val="2"/>
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>
    <c:plotArea>
      <c:layout/>
      <c:lineChart>
        <c:grouping val="standard"/>
{primary_series_xml}
        <c:axId val="123456"/>
        <c:axId val="654321"/>
      </c:lineChart>
      <c:lineChart>
        <c:grouping val="standard"/>
{secondary_series_xml}
        <c:axId val="{secondary_x_axis_id}"/>
        <c:axId val="{secondary_y_axis_id}"/>
      </c:lineChart>
      <c:catAx><c:axId val="123456"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="b"/>{x_axis_title}{x_axis_num_format}<c:majorTickMark val="cross"/><c:minorTickMark val="cross"/><c:crossAx val="654321"/><c:tickLblPos val="nextTo"/>{text_rotation}</c:catAx>
      <c:valAx><c:axId val="654321"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="l"/>{y_axis_title}{primary_y_axis_format}<c:majorTickMark val="cross"/><c:minorTickMark val="cross"/><c:crossAx val="123456"/><c:tickLblPos val="nextTo"/><c:crosses val="autoZero"/><c:crossBetween val="between"/></c:valAx>
      <c:valAx><c:axId val="{secondary_y_axis_id}"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="r"/>{secondary_y_axis_title}{secondary_y_axis_format}<c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:crossAx val="{secondary_x_axis_id}"/><c:tickLblPos val="nextTo"/><c:crosses val="max"/><c:crossBetween val="between"/></c:valAx>
      <c:catAx><c:axId val="{secondary_x_axis_id}"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="1"/><c:axPos val="b"/><c:numFmt formatCode="General" sourceLinked="1"/><c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/><c:crossAx val="{secondary_y_axis_id}"/><c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/><c:noMultiLvlLbl val="0"/></c:catAx>
    </c:plotArea>
    {legend}
    <c:plotVisOnly val="1"/>
    <c:dispBlanksAs val="zero"/>
    <c:showDLblsOverMax val="1"/>
  </c:chart>
</c:chartSpace>"""


def _scatter_series_xml(index: int, series: ChartSeries) -> str:
    name, x_ref, y_ref, error_ref = series
    error_bars = ""
    if error_ref:
        error_bar_colors = ("C00000", "0070C0")
        error_bar_line = ""
        if index < len(error_bar_colors):
            error_bar_line = f'<c:spPr><a:ln><a:solidFill><a:srgbClr val="{error_bar_colors[index]}"/></a:solidFill></a:ln></c:spPr>'
        error_bars = f"""
          <c:errBars>
            <c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/>
            {error_bar_line}
            <c:plus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:plus>
            <c:minus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:minus>
          </c:errBars>"""
    return f"""
        <c:ser>
          <c:idx val="{index}"/><c:order val="{index}"/>
          <c:tx><c:v>{escape(name)}</c:v></c:tx>
          <c:marker><c:symbol val="circle"/></c:marker>
{error_bars}
          <c:xVal><c:numRef><c:f>{escape(x_ref)}</c:f></c:numRef></c:xVal>
          <c:yVal><c:numRef><c:f>{escape(y_ref)}</c:f></c:numRef></c:yVal>
        </c:ser>"""


def _column_series_xml(index: int, series: ChartSeries) -> str:
    name, categories_ref, values_ref, _ = series[:4]
    fill_color = series[4] if len(series) >= 5 else None
    fill = f'<c:spPr><a:solidFill><a:srgbClr val="{fill_color}"/></a:solidFill></c:spPr>' if fill_color else ""
    return f"""
        <c:ser>
          <c:idx val="{index}"/><c:order val="{index}"/>
          <c:tx><c:v>{escape(name)}</c:v></c:tx>
{fill}
          <c:cat><c:strRef><c:f>{escape(categories_ref)}</c:f></c:strRef></c:cat>
          <c:val><c:numRef><c:f>{escape(values_ref)}</c:f></c:numRef></c:val>
        </c:ser>"""


def _chart_series_xml(index: int, series: ChartSeries) -> str:
    name, categories_ref, values_ref, error_ref = series[:4]
    line_width = series[4] if len(series) >= 5 else None
    line_color = series[5] if len(series) >= 6 else None
    marker_symbol = series[6] if len(series) >= 7 else "circle"
    dash_style = series[7] if len(series) >= 8 else None
    category_ref_type = series[8] if len(series) >= 9 else "str"
    line_attrs = f' w="{line_width}"' if line_width else ""
    line_fill = f'<a:solidFill><a:srgbClr val="{line_color}"/></a:solidFill>' if line_color else ""
    dash = f'<a:prstDash val="{dash_style}"/>' if dash_style else ""
    line_style = f'<c:spPr><a:ln{line_attrs}>{line_fill}{dash}</a:ln></c:spPr>' if line_width or line_color or dash_style else ""
    if category_ref_type == "num":
        category_ref = f'<c:numRef><c:f>{escape(categories_ref)}</c:f></c:numRef>'
    else:
        category_ref = f'<c:strRef><c:f>{escape(categories_ref)}</c:f></c:strRef>'
    error_bars = ""
    if error_ref:
        error_bar_colors = ("C00000", "0070C0")
        error_bar_line = ""
        if index < len(error_bar_colors):
            error_bar_line = f'<c:spPr><a:ln><a:solidFill><a:srgbClr val="{error_bar_colors[index]}"/></a:solidFill></a:ln></c:spPr>'
        error_bars = f"""
          <c:errBars>
            <c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/>
            {error_bar_line}
            <c:plus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:plus>
            <c:minus><c:numRef><c:f>{escape(error_ref)}</c:f></c:numRef></c:minus>
          </c:errBars>"""
    return f"""
        <c:ser>
          <c:idx val="{index}"/><c:order val="{index}"/>
          <c:tx><c:v>{escape(name)}</c:v></c:tx>
          <c:marker><c:symbol val="{escape(marker_symbol)}"/></c:marker>
{line_style}
{error_bars}
          <c:cat>{category_ref}</c:cat>
          <c:val><c:numRef><c:f>{escape(values_ref)}</c:f></c:numRef></c:val>
          <c:smooth val="0"/>
        </c:ser>"""
