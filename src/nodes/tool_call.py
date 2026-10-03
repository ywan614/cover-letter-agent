"""确定性地规范信件 Markdown，并用固定样式生成 PDF。"""
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate

from ..config import PROJECT_ROOT
from ..state import AgentState


def normalize_markdown(value: str) -> str:
    """仅修正表示方式，不猜测信件结构、不改写或删除正文。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_markdown 必须是非空字符串")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    fenced = re.fullmatch(r"```(?:markdown|md)?\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise ValueError("求职信正文不能为空")
    if any(unicodedata.category(c).startswith("C") and c != "\n" for c in text):
        raise ValueError("求职信含不支持的控制字符")
    # 固定的纯文本子集：未支持的语法应失败，不能悄悄当正文打印。
    if re.search(
        r"(?m)^\s*(?:#{1,6}\s|>|[-+*]\s|\d+[.)]\s|[-=_*~]{3,}\s*$|\|)"
        r"|[`*]|(?<!\w)_(?=\S)|(?<=\S)_(?!\w)|!?\[[^\]\n]*\]"
        r"|<\s*/?\s*[A-Za-z][^>]*>|(?m:^[^\n]*\|[^\n]*\n\s*:?-{3,})",
        text,
    ):
        raise ValueError("求职信只支持纯文本段落和硬换行，不支持其他 Markdown 或 HTML 语法")
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if line.endswith("\\"):
                line = line[:-1].rstrip()
            if "\\" in line or not line:
                raise ValueError("求职信含不支持的转义或空行")
            lines.append(line)
        blocks.append("  \n".join(lines))
    return "\n\n".join(blocks) + "\n"


def render_pdf(markdown: str, path: Path) -> None:
    """使用包内字体，避免依赖宿主机器的字体；长信自然分页。"""
    font_name = "CoverLetterVera"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        font_path = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    font = pdfmetrics.getFont(font_name)
    if any(ord(c) not in font.face.charToGlyph for c in markdown if not c.isspace()):
        raise ValueError("求职信包含 PDF 字体不支持的字符")
    style = ParagraphStyle(
        "Letter", fontName=font_name, fontSize=11, leading=15,
        alignment=TA_LEFT, spaceAfter=10, allowWidows=0, allowOrphans=0,
    )
    paragraphs = [
        Paragraph("<br/>".join(escape(line.rstrip()) for line in block.splitlines()), style)
        for block in markdown.strip().split("\n\n")
    ]
    # 按格式契约最后一段为落款；保持该段完整，不强塞整封信到一页。
    story = paragraphs[:-1] + [KeepTogether([paragraphs[-1]])]
    SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=22 * mm, bottomMargin=22 * mm, title="Cover Letter",
    ).build(story)


def save_output(state: AgentState, *, output_dir: Path | None = None) -> dict:
    """两份文件生成成功后发布；异常向上传递，不报告成功路径。"""
    markdown = normalize_markdown(state["output_markdown"])
    if output_dir is None:
        root = PROJECT_ROOT / "output"
        root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "cover_letter.md"
    pdf_path = output_dir / "cover_letter.pdf"
    with TemporaryDirectory(prefix=".save-", dir=output_dir) as temporary:
        draft_md = Path(temporary) / markdown_path.name
        draft_pdf = Path(temporary) / pdf_path.name
        draft_md.write_text(markdown, encoding="utf-8")
        render_pdf(markdown, draft_pdf)
        # 单文件替换是原子的；两个文件不是整体事务，失败后可重跑本节点。
        draft_md.replace(markdown_path)
        draft_pdf.replace(pdf_path)
    return {
        "output_markdown": markdown,
        "output_files": {"markdown_path": str(markdown_path), "pdf_path": str(pdf_path)},
    }
