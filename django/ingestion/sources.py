"""Arxiv Troller discovery, arXiv metadata and version-pinned PDF downloads."""
import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests

from .pdf_text import pdf_pages

ARXIV_ID = re.compile(r"(?<![\w.])((?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7}))(?:v\d+)?", re.I)


ARXIV_DOI = re.compile(r"^/?10\.48550/arxiv\.", re.I)


def arxiv_id(value):
    if not value:
        return None
    if "://" in value:
        url = urlparse(value)
        if url.hostname in {"doi.org", "dx.doi.org"} and ARXIV_DOI.match(url.path):
            value = ARXIV_DOI.sub("", url.path)  # arXiv DOIs: 10.48550/arXiv.<id>
        elif url.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            return None
    match = ARXIV_ID.search(value)
    return match[1] if match else None


def fetch_metadata(identifier, session=None):
    """Title, abstract and dates from the arXiv API, in Arxiv Troller's field names."""
    session = session or requests.Session()
    response = session.get("https://export.arxiv.org/api/query", params={"id_list": identifier}, timeout=60)
    response.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(response.content).find("a:entry", ns)
    if entry is None or arxiv_id(entry.findtext("a:id", "", ns)) != identifier:
        raise ValueError(f"arXiv has no entry for {identifier}")
    text = lambda tag: " ".join(entry.findtext(f"a:{tag}", "", ns).split())
    comment = " ".join(entry.findtext("{http://arxiv.org/schemas/atom}comment", "").split())
    return {"arxiv_id": identifier, "title": text("title"), "abstract": text("summary"),
            "created": text("published"), "updated": text("updated"), "comment": comment}


MAX_CURSOR = 3500
logger = logging.getLogger(__name__)


class Troller:
    def __init__(self, url, account, session=None):
        self.url = url.rstrip("/")
        self.account = account
        self.session = session or requests.Session()

    def login(self):
        response = self.session.get(self.url + "/", timeout=30)
        response.raise_for_status()
        response = self.session.post(self.url + "/login/", data={"email": self.account},
                                     headers=self._headers(), timeout=30)
        response.raise_for_status()
        # Confirm login and the integration endpoint before attempting any changes.
        self.request("tags")

    def _headers(self):
        return {"X-CSRFToken": self.session.cookies.get("csrftoken", ""), "Referer": self.url + "/"}

    def request(self, action, *, write=False, **params):
        url = self.url + "/api/ingestion/"
        params["action"] = action
        if write:
            response = self.session.post(url, json=params, headers=self._headers(), timeout=60)
        else:
            response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise ValueError(payload.get("error", "Arxiv Troller request failed"))
        return payload

    def papers(self, action, **params):
        # Cursors are offsets for tag/similar and opaque strings for search; pass them back as given.
        cursor, seen = None, set()
        for _ in range(100):
            page = self.request(action, **params, **({"cursor": cursor} if cursor is not None else {}))
            yield from page["papers"]
            cursor = page.get("next_cursor")
            if cursor is None:
                return
            if isinstance(cursor, str) and len(cursor) > MAX_CURSOR:
                # Troller's search cursor lists every paper returned so far; near its 400-result
                # cap the GET request line exceeds gunicorn's 4,094-byte limit. Stop here.
                logger.warning("Stopping %s at a %d-character cursor", action, len(cursor))
                return
            if cursor in seen:
                raise ValueError("Arxiv Troller pagination did not advance")
            seen.add(cursor)
        raise ValueError("Arxiv Troller pagination limit reached; narrow the search")

    def tag_search(self, tag, since):
        """The site's joint similarity search over a tag's papers (up to 400 results)."""
        return self.papers("search", type="tag", tag=tag, since=since)

    def sync(self, source_tag, target_tag, ids):
        self.request("copy_tag", write=True, source=source_tag, target=target_tag)
        missing = []
        for offset in range(0, len(ids), 200):
            result = self.request("bulk_add", write=True, tag=target_tag, arxiv_ids=ids[offset:offset + 200])
            missing.extend(result["missing"])
        return missing


def fetch_pdf(identifier, session=None):
    """(version, PDF bytes, page texts) for the current version, within the size limits."""
    session = session or requests.Session()
    response = session.get("https://export.arxiv.org/api/query", params={"id_list": identifier}, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    entry_id = entry.findtext("a:id", "", ns) if entry is not None else ""
    match = re.search(r"v(\d+)$", entry_id)
    if arxiv_id(entry_id) != identifier or not match:
        raise ValueError("arXiv did not return an explicit version for this paper")
    version = int(match[1])
    url = f"https://arxiv.org/pdf/{identifier}v{version}"
    with session.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > 30 * 1024 * 1024:
                raise ValueError("Paper exceeds the 30 MB download limit")
    data = bytes(content)
    if not data.startswith(b"%PDF-"):
        raise ValueError("arXiv returned a non-PDF response")
    pages = pdf_pages(data, max_pages=200)
    if sum(map(len, pages)) > 350_000:
        raise ValueError("Paper exceeds the text budget; requires a separate extraction")
    if not any(page.strip() for page in pages):
        raise ValueError("PDF has no extractable text; requires visual inspection")
    return version, data, pages
