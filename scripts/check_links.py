from __future__ import annotations

import argparse
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "a" and "href" in attr_map:
            self.links.append(attr_map["href"] or "")
        elif tag in {"img", "script"} and "src" in attr_map:
            self.links.append(attr_map["src"] or "")
        elif tag == "link" and "href" in attr_map:
            self.links.append(attr_map["href"] or "")


def _is_external(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}


def _normalize_target(url: str, base_dir: Path, site_root: Path) -> Path | None:
    if not url or url.startswith("#"):
        return None

    if url.startswith("mailto:") or url.startswith("tel:"):
        return None

    if _is_external(url):
        return None

    url = url.split("#", 1)[0].split("?", 1)[0]
    if not url:
        return None

    if url.startswith("/"):
        # Strip site base path if present.
        url = url.lstrip("/")
        if url.startswith("ivrobust/"):
            url = url[len("ivrobust/") :]
        target = site_root / url
    else:
        target = base_dir / url

    if target.is_dir() or str(target).endswith("/"):
        target = target / "index.html"

    return target


def check_links(site_dir: Path) -> list[str]:
    failures: list[str] = []
    html_files = list(site_dir.rglob("*.html"))
    for html_path in html_files:
        parser = LinkParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        base_dir = html_path.parent

        for link in parser.links:
            target = _normalize_target(link, base_dir=base_dir, site_root=site_dir)
            if target is None:
                continue
            if not target.exists():
                rel_src = html_path.relative_to(site_dir)
                rel_target = os.path.relpath(target, site_dir)
                failures.append(f"{rel_src} -> {link} (missing {rel_target})")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check internal links in docs")
    parser.add_argument(
        "--site-dir",
        default="docs",
        help="Directory containing built HTML (default: docs)",
    )
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    if not site_dir.exists():
        raise SystemExit(f"Site directory not found: {site_dir}")

    failures = check_links(site_dir)
    if failures:
        print("Broken internal links detected:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"Link check passed for {len(list(site_dir.rglob('*.html')))} HTML files.")


if __name__ == "__main__":
    main()
