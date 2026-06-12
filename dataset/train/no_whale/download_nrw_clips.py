#!/usr/bin/env python3
"""
Download confirmed audio clips for TSS-AB01 from portal.nrwbuoys.org.

Usage:
    python download_nrw_clips.py [options]

Requires:
    pip install requests beautifulsoup4

Examples:
    # Download first 1000 confirmed clips for TSS-AB01 (default)
    python download_nrw_clips.py

    # Different buoy or limit
    python download_nrw_clips.py --position TSS-AB02 --limit 500

    # Save as MP3 instead of WAV
    python download_nrw_clips.py --format mp3

    # Dump the raw list-page HTML for debugging
    python download_nrw_clips.py --debug
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://portal.nrwbuoys.org"
LIST_URL   = BASE_URL + "/ab/clips/rejected/"
WAV_URL    = BASE_URL + "/ab/clips/as_wav/{clip_id}/play.wav"
MP3_URL    = BASE_URL + "/ab/clips/as_mp3/{clip_id}/play.mp3"
DELAY      = 0.4   # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── List page scraping ────────────────────────────────────────────────────────

def get_clip_ids_from_page(session, url):
    """
    Fetch one page of the confirmed clips list and return:
      (list_of_int_clip_ids, next_page_url_or_None)

    The list page shows rows like:
      <a href="/ab/clips/818302">818302</a>  ...  TSS-AB01  ...
    We collect all /ab/clips/{id} hrefs and their row text to filter by position.
    """
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        sys.exit(
            f"ERROR: list page returned HTTP {resp.status_code}.\n"
            f"URL: {url}\n"
            f"Try --debug to inspect the raw HTML.\n"
            f"If prompted for login, the site may require authentication."
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    # Collect all clip links: /ab/clips/<numeric_id>
    clip_ids = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parts = href.rstrip("/").split("/")
        if (len(parts) >= 3
                and parts[-2] == "clips"
                and parts[-1].isdigit()):
            clip_ids.append(int(parts[-1]))

    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    # Find the "next" pagination link
    next_url = None
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        if text in ("next", "next »", "›", ">", "next page"):
            next_url = urljoin(url, a["href"])
            break

    return unique_ids, next_url


def collect_all_clip_ids(session, position, limit, debug):
    """Page through the confirmed clips list, filtered to `position`, up to `limit`."""
    url = LIST_URL + f"?position={position}"
    all_ids = []
    page_num = 1

    while url and len(all_ids) < limit:
        print(f"  Fetching list page {page_num}: {url}")
        ids, next_url = get_clip_ids_from_page(session, url)

        if debug and page_num == 1:
            resp = session.get(url, timeout=30)
            out_path = Path("debug_list_page.html")
            out_path.write_text(resp.text, encoding="utf-8")
            print(f"  [debug] Saved list page HTML → {out_path}")

        if not ids:
            print("  (no clip IDs found on this page — stopping pagination)")
            break

        all_ids.extend(ids)
        print(f"  Found {len(ids)} clip(s) on page {page_num}  "
              f"(total so far: {len(all_ids)})")

        page_num += 1
        url = next_url
        if url:
            time.sleep(DELAY)

    return all_ids[:limit]


# ── Download ──────────────────────────────────────────────────────────────────

def download_clip(session, clip_id, dest_dir, fmt, overwrite):
    """
    Download one audio clip. Returns 'downloaded', 'skipped', or 'missing'.
    """
    template = WAV_URL if fmt == "wav" else MP3_URL
    url = template.format(clip_id=clip_id)
    filename = f"{clip_id}.{fmt}"
    dest = dest_dir / filename

    if dest.exists() and not overwrite:
        return "skipped"

    resp = session.get(url, stream=True, timeout=60)

    if resp.status_code == 404:
        return "missing"

    resp.raise_for_status()

    size = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)
                size += len(chunk)

    return f"{size / 1024:.1f} KB"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download confirmed NRW buoy audio clips for a given position.")
    parser.add_argument("-p", "--position", default="TSS-AB01",
                        help="Buoy position name (default: TSS-AB01)")
    parser.add_argument("-n", "--limit", type=int, default=500,
                        help="Maximum clips to download (default: 1000)")
    parser.add_argument("-o", "--output", default=".",
                        help="Output directory (default: current folder)")
    parser.add_argument("-f", "--format", choices=["wav", "mp3"], default="mp3",
                        help="Audio format to download (default: mp3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download files that already exist on disk")
    parser.add_argument("--debug", action="store_true",
                        help="Save the first list page HTML for inspection")
    args = parser.parse_args()

    dest_dir = Path(args.output)
    dest_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"\nCollecting clip IDs for position={args.position} "
          f"(limit={args.limit}) …\n")
    clip_ids = collect_all_clip_ids(
        session, args.position, args.limit, args.debug)

    if not clip_ids:
        print("\nNo clip IDs were found.")
        print("Tips:")
        print("  • Run with --debug to inspect the raw HTML")
        print("  • Try --position TSS-AB01 (note two S's)")
        print("  • Confirm you can open the URL in a browser:")
        print(f"    {LIST_URL}?position={args.position}")
        sys.exit(1)

    total   = len(clip_ids)
    done    = 0
    skipped = 0
    missing = 0

    print(f"\nDownloading {total} clip(s) → {dest_dir.resolve()}\n")

    for i, cid in enumerate(clip_ids, 1):
        result = download_clip(session, cid, dest_dir, args.format, args.overwrite)

        if result == "skipped":
            skipped += 1
            status = "skip"
        elif result == "missing":
            missing += 1
            status = "N/A "
        else:
            done += 1
            status = f"✓   ({result})"

        # Show progress every clip, with a counter
        print(f"  [{i:>4}/{total}]  {cid}.{args.format}  {status}")
        time.sleep(DELAY)

    print(f"\n{'─'*50}")
    print(f"Done.  Downloaded: {done}  |  Skipped: {skipped}  |  Not available: {missing}")
    print(f"Files saved to: {dest_dir.resolve()}")


if __name__ == "__main__":
    main()
