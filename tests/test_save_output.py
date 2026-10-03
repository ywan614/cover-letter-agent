"""离线验证格式修正、实际 PDF 内容、分页和失败边界。"""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from src.nodes.tool_call import normalize_markdown, save_output
from src.state import create_initial_state


LETTER = (
    "Alex Example\nExample City\nalex@example.invalid\n"
    "23 September 2026\n\nDear Hiring Manager,\n\n"
    "I built R&D systems with Python & SQL, improving latency by < 5%. "
    "My team's tools support end-to-end delivery.\n\nKind regards,\nAlex Example"
)


class SaveOutputTest(unittest.TestCase):
    def state(self, letter=LETTER):
        return {**create_initial_state(jd_text="AI Engineer"), "output_markdown": letter}

    def test_normalization_and_real_pdf(self):
        raw = "```markdown\r\n" + LETTER.replace("\n", "\r\n") + "\r\n```"
        normalized = normalize_markdown(raw)
        self.assertIn("Alex Example  \nExample City", normalized)
        self.assertEqual(normalize_markdown(normalized), normalized)
        self.assertEqual(normalize_markdown("Kind regards,\\\nAlex"), "Kind regards,  \nAlex\n")
        with tempfile.TemporaryDirectory() as directory:
            state = self.state(raw)
            result = save_output(state, output_dir=Path(directory))
            self.assertEqual(state["output_markdown"], raw)
            self.assertEqual(state["output_files"], {"markdown_path": None, "pdf_path": None})
            self.assertEqual(result["output_markdown"], normalized)
            paths = result["output_files"]
            self.assertEqual(Path(paths["markdown_path"]).read_text(), normalized)
            reader = PdfReader(paths["pdf_path"])
            self.assertEqual(len(reader.pages), 1)
            self.assertAlmostEqual(float(reader.pages[0].mediabox.width), 595.28, places=1)
            text = reader.pages[0].extract_text()
            self.assertEqual(" ".join(text.split()), " ".join(LETTER.split()))
            self.assertTrue(all(Path(p).is_absolute() for p in paths.values()))
            self.assertEqual(save_output(self.state(raw), output_dir=Path(directory)), result)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_rejected_inputs(self):
        for value in (None, " ", "```md\n\n```", "# Heading", "**bold**", "_italic_",
                      "[link](https://example.com)", "<b>HTML</b>", "- list", "1. list",
                      "a | b\n--- | ---", "text\x00", "~~~\ncode\n~~~"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_markdown(value)

    def test_render_failure_preserves_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cover_letter.md", "cover_letter.pdf"):
                (root / name).write_bytes(b"previous")
            with patch("src.nodes.tool_call.render_pdf", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    save_output(self.state(), output_dir=root)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["cover_letter.md", "cover_letter.pdf"])
            self.assertTrue(all(p.read_bytes() == b"previous" for p in root.iterdir()))
            with self.assertRaisesRegex(ValueError, "字体"):
                save_output(self.state("Hello 😀"), output_dir=root)

    def test_long_letter_pages_without_loss(self):
        body = "\n\n".join(f"Paragraph {i}: " + "Experience with Python and SQL. " * 18 for i in range(18))
        letter = body + "\n\nKind regards,\nAlex Example"
        with tempfile.TemporaryDirectory() as directory:
            result = save_output(self.state(letter), output_dir=Path(directory))
            pages = PdfReader(result["output_files"]["pdf_path"]).pages
            self.assertGreater(len(pages), 1)
            text = " ".join(page.extract_text() for page in pages)
            self.assertEqual(re.sub(r"\s+", " ", text).strip(), " ".join(letter.split()))
            self.assertIn("Kind regards,", pages[-1].extract_text())
            self.assertIn("Alex Example", pages[-1].extract_text())


if __name__ == "__main__":
    unittest.main()
