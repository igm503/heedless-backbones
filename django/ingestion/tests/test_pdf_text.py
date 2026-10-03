from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from ingestion.pdf_text import page_text, pdf_pages


def word(text, x0, x1, top, bottom, size=9.0):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "size": size}


def page(words, width=600):
    fake = MagicMock(width=width)
    fake.extract_words.return_value = words
    return fake


class PdfTextTests(SimpleTestCase):
    def test_table_cells_stay_apart_and_superscripts_attach(self):
        # From ConvNeXt Table 1, where pypdf produced "ConvNeXt-T 224229M 4.5G".
        words = [word("ConvNeXt-T", 57.2, 104.2, 173.1, 182.0), word("224", 129.7, 143.2, 173.1, 182.0),
                 word("2", 143.19, 146.8, 171.5, 177.4, size=6.0), word("29M", 157.5, 174.5, 173.1, 182.0),
                 word("4.5G", 182.0, 199.0, 173.1, 182.0),
                 # The next row's superscript sits just below this row; it must not join it.
                 word("ConvNeXt-S", 57.2, 104.2, 184.1, 193.0), word("224", 129.7, 143.2, 184.1, 193.0),
                 word("2", 143.19, 146.8, 182.4, 188.4, size=6.0)]
        lines = page_text(page(words)).splitlines()
        self.assertEqual(lines[0], "ConvNeXt-T   224^2 29M 4.5G")
        self.assertEqual(lines[1], "ConvNeXt-S   224^2")

    def test_two_columns_are_read_in_order_around_full_width_lines(self):
        words = [word("Title", 250, 350, 10, 20)]
        for row in range(4):
            y = 40 + 12 * row
            words += [word(f"left{row}", 50, 280, y, y + 9), word(f"right{row}", 320, 550, y, y + 9)]
        words.append(word("Wide table row across the page", 60, 540, 100, 109))
        words += [word("left4", 50, 280, 120, 129), word("right4", 320, 550, 120, 129)]
        self.assertEqual(page_text(page(words)).splitlines(), [
            "Title", "left0", "left1", "left2", "left3", "right0", "right1", "right2", "right3",
            "Wide table row across the page", "left4", "right4"])

    def test_single_column_pages_are_left_alone(self):
        words = [word("A full line of body text", 50, 550, 40 + 12 * row, 49 + 12 * row) for row in range(5)]
        self.assertEqual(len(page_text(page(words)).splitlines()), 5)

    def test_page_limit(self):
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [page([])] * 3
        with patch("ingestion.pdf_text.pdfplumber.open", return_value=pdf):
            self.assertEqual(pdf_pages(b"%PDF", max_pages=5), ["", "", ""])
            with self.assertRaisesRegex(ValueError, "2-page"):
                pdf_pages(b"%PDF", max_pages=2)

    def test_control_characters_are_removed(self):
        # A figure font without a Unicode mapping (arXiv 2610.01403, p. 16) yields glyph codes
        # as UTF-16 bytes, NULs included, which Postgres cannot store.
        glyphs = "\x006\x00R\x00I\x00W\x00P\x00D\x00[\x00\x10\x00I\x00U\x00H\x00H"
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [page([word("Figure", 50, 90, 40, 49), word(glyphs, 100, 300, 40, 49),
                                                  word("caption", 50, 90, 60, 69)])]
        with patch("ingestion.pdf_text.pdfplumber.open", return_value=pdf):
            [text] = pdf_pages(b"%PDF")
        self.assertEqual(text, "Figure 6RIWPD[IUHH\ncaption")
