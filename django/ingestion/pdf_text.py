"""Page text from word positions, so that citations can be checked against it.

Text extractors that stream characters (pypdf) run table cells together and flatten
superscripts ("224² | 29M" -> "224229M"). This rebuilds lines from pdfplumber's word
boxes: table cells stay separated, superscripts are written with a caret (224^2),
and two-column pages are read column by column, with full-width lines (titles, wide
tables, captions) kept whole.
"""
import io
import re
import statistics

import pdfplumber

SMALL = 0.85  # Text below this fraction of the body size is a superscript or subscript.
LINE_TOLERANCE = 2.5  # Points between baselines of words on the same line.
TOUCHING = 2  # Points between a word and the superscript attached to it.
# Control characters other than tab and newline. A font without a Unicode mapping can yield
# raw glyph codes, NULs among them, which Postgres cannot store in the pages JSON.
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def is_small(word, size):
    return word["size"] < size * SMALL


def lines_of(words):
    """Group words into lines by baseline. Small raised or lowered words join the line
    whose text they touch, instead of forming lines of their own."""
    body = statistics.median(word["size"] for word in words)
    lines = []

    def line_at(bottom):
        return next((line for line in lines if abs(line["bottom"] - bottom) <= LINE_TOLERANCE), None)

    def new_line(word):
        lines.append({"bottom": word["bottom"], "top": word["top"], "size": word["size"], "words": [word]})

    for word in sorted((w for w in words if not is_small(w, body)), key=lambda w: (w["bottom"], w["x0"])):
        line = line_at(word["bottom"])
        if line:
            line["words"].append(word)
            line["top"] = min(line["top"], word["top"])
        else:
            new_line(word)
    for word in (w for w in words if is_small(w, body)):
        middle = (word["top"] + word["bottom"]) / 2
        hosts = [line for line in lines
                 if line["size"] > word["size"] and line["top"] - 1 <= middle <= line["bottom"] + 1
                 and any(-1 <= word["x0"] - other["x1"] < TOUCHING for other in line["words"])]
        if hosts:
            min(hosts, key=lambda line: abs((line["top"] + line["bottom"]) / 2 - middle))["words"].append(word)
        elif line := line_at(word["bottom"]):
            line["words"].append(word)
        else:
            new_line(word)
    return sorted(lines, key=lambda line: line["bottom"])


def render(line):
    """A line's words left to right; wide gaps (between table cells) become three spaces."""
    text, previous = "", None
    for word in sorted(line["words"], key=lambda w: w["x0"]):
        small = is_small(word, line["size"])
        raised = small and word["bottom"] < line["bottom"] - 1
        touching = previous is not None and word["x0"] - previous["x1"] < TOUCHING
        if previous is None:
            text = ("^" if raised else "") + word["text"]
        elif small and touching:
            text += ("^" if raised else "") + word["text"]
        else:
            gap = word["x0"] - previous["x1"]
            text += ("   " if gap > line["size"] * 1.5 else " ") + word["text"]
        previous = word
    return text


def crosses(line, x):
    return any(word["x0"] - 1 < x < word["x1"] + 1 for word in line["words"])


def gutter(lines, width):
    """The x position between 35% and 65% of the page width that the fewest lines cross."""
    candidates = [width * (0.35 + 0.3 * step / 70) for step in range(71)]
    counts = {x: sum(crosses(line, x) for line in lines) for x in candidates}
    x = min(candidates, key=counts.get)
    return x, counts[x]


def page_text(page):
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2, use_text_flow=False, extra_attrs=["size"])
    if not words:
        return ""
    lines = lines_of(words)
    split, crossing = gutter(lines, page.width)
    left_lines = sum(any(word["x1"] <= split for word in line["words"]) for line in lines)
    right_lines = sum(any(word["x0"] >= split for word in line["words"]) for line in lines)
    # Two columns: most lines stay on one side of the gutter, and both sides have text.
    if crossing > 0.3 * len(lines) or min(left_lines, right_lines) < 0.2 * len(lines):
        return "\n".join(render(line) for line in lines)
    out, left, right = [], [], []
    for line in lines:
        if crosses(line, split):
            # A full-width line (title, wide table row, caption) ends a two-column section.
            out += [render(part) for part in left + right] + [render(line)]
            left, right = [], []
            continue
        for column, part in ((left, [w for w in line["words"] if w["x1"] <= split]),
                             (right, [w for w in line["words"] if w["x0"] >= split])):
            if part:
                column.append({**line, "words": part})
    out += [render(part) for part in left + right]
    return "\n".join(out)


def pdf_links(data, first_pages=2):
    """Hyperlink targets on the first pages (code and project links are usually here)."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return sorted({link["uri"] for page in pdf.pages[:first_pages] for link in page.hyperlinks if link.get("uri")})


def pdf_pages(data, max_pages=None):
    """Text of each page of a PDF given as bytes."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        if max_pages is not None and len(pdf.pages) > max_pages:
            raise ValueError(f"Paper exceeds the {max_pages}-page extraction limit")
        return [CONTROL.sub("", page_text(page)) for page in pdf.pages]
