"""Official web sources for agent extractions: the paper's repository, its releases, and its
project page. Only pages the paper itself links to (on its first two pages, or in its arXiv
abstract or comments) may be fetched, and each is saved so citations can be re-checked."""
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests
from django.utils import timezone

URL = re.compile(r"https?://[^\s<>\"')\]]+|(?<![\w/.])(?:www\.)?github\.com/[^\s<>\"')\]]+", re.I)
GITHUB_REPO = re.compile(r"^/([\w.-]+)/([\w.-]+)")
IGNORED_HOSTS = {"arxiv.org", "doi.org", "dx.doi.org", "creativecommons.org", "openreview.net"}
MAX_BYTES = 5 * 1024 * 1024


def repo_of(url):
    """(owner, repo) for a github.com URL, else None."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return None
    match = GITHUB_REPO.match(parsed.path)
    return (match[1].lower(), re.sub(r"\.git$", "", match[2]).lower()) if match else None


def urls_in(text):
    # PDF text often breaks long URLs across lines, sometimes at a hyphen that belongs to the
    # name and sometimes with one added; try the text as is, joined, and de-hyphenated.
    joined = re.sub(r"\s*\n\s*", "", text)
    dehyphenated = re.sub(r"-\s*\n\s*", "", text)
    return {match.rstrip(".,;:") for variant in (text, joined, dehyphenated) for match in URL.findall(variant)}


def official_links(pages, links=(), metadata=None):
    """Repositories and project hosts linked from the paper's first two pages or arXiv metadata."""
    candidates = set(links)
    for text in pages[:2]:
        candidates |= urls_in(text)
    for field in ("abstract", "comment"):
        candidates |= urls_in((metadata or {}).get(field) or "")
    repos, hosts = set(), set()
    for url in candidates:
        repo = repo_of(url)
        if repo:
            repos.add(repo)
            continue
        host = urlparse(url if "://" in url else "https://" + url).hostname or ""
        if host and host not in IGNORED_HOSTS and "." in host:
            hosts.add(host.lower())
    return repos, hosts


def check_allowed(url, repos, hosts):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs can be fetched")
    host = (parsed.hostname or "").lower()
    if host in {"github.com", "www.github.com"}:
        if repo_of(url) not in repos:
            raise ValueError(f"{url} is not the paper's repository (linked: {sorted('/'.join(r) for r in repos)})")
    elif host == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or (parts[0].lower(), parts[1].lower()) not in repos:
            raise ValueError(f"{url} is not in the paper's repository")
    elif host not in hosts:
        raise ValueError(f"{url} is not linked from the paper (linked hosts: {sorted(hosts)})")


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "pre"}

    def __init__(self):
        super().__init__()
        self.parts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def html_text(markup):
    parser = TextExtractor()
    parser.feed(markup)
    lines = (" ".join(line.split()) for line in html.unescape("".join(parser.parts)).splitlines())
    return "\n".join(line for line in lines if line)


def download(url, session, headers=None):
    with session.get(url, headers=headers or {}, timeout=(15, 60), stream=True) as response:
        response.raise_for_status()
        content = b""
        for chunk in response.iter_content(65536):
            content += chunk
            if len(content) > MAX_BYTES:
                raise ValueError("Source exceeds the 5 MB limit")
        return content, response.headers.get("content-type", "")


def fetch(url, session=None):
    """(fetched_from, text, sha256) for a source URL, reading GitHub through its API or raw files."""
    session = session or requests.Session()
    parsed = urlparse(url)
    repo = repo_of(url)
    path = parsed.path.strip("/").split("/")
    if repo and len(path) >= 3 and path[2] == "releases":
        source = f"https://api.github.com/repos/{repo[0]}/{repo[1]}/releases?per_page=100"
        content, _ = download(source, session, {"Accept": "application/vnd.github+json"})
        releases = json.loads(content)
        text = "\n\n".join(f"# {release.get('name') or release.get('tag_name')}\n{release.get('body') or ''}"
                           for release in releases)
    elif repo and len(path) >= 5 and path[2] in {"blob", "tree"}:
        source = f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{'/'.join(path[3:])}"
        content, _ = download(source, session)
        text = content.decode("utf-8", errors="replace")
    elif repo:
        source = f"https://api.github.com/repos/{repo[0]}/{repo[1]}/readme"
        content, _ = download(source, session, {"Accept": "application/vnd.github.raw"})
        text = content.decode("utf-8", errors="replace")
    else:
        source = url
        content, kind = download(url, session)
        text = content.decode("utf-8", errors="replace")
        if "html" in kind or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            text = html_text(text)
    return source, text, hashlib.sha256(content).hexdigest()


def snapshot(url, pages, links=(), metadata=None, session=None):
    repos, hosts = official_links(pages, links, metadata)
    check_allowed(url, repos, hosts)
    fetched_from, text, digest = fetch(url, session)
    return {"url": url, "fetched_from": fetched_from, "fetched_at": timezone.now().isoformat(),
            "sha256": digest, "text": text}
