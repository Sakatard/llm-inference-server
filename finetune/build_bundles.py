"""Phase 4 step 1: build TeacherContextBundles for N substantive Polymarket markets.

Sources from Gamma (closed=true, age 0-180d, binary YES/NO, volume ≥ $1k,
substantive question text, non-microstructure). For each market:
- market_payload = the Gamma response dict
- parsed_rules = polymarket-agents `parse_market_rules(market)`
- connector_summary_redacted = {"news": [...], "data_quality": ...} where news
  comes from gnews (decision_ts - 7d, decision_ts] window, fallback gdelt
- intended_size_usd = 500

Phase 3 finding: ~40% of substantive markets have news_count ≥ 5. We
oversample (3× target) and filter to those that pass the news bar, mirroring
the SPEC's "partial-context training" contingency.

Writes one JSON object per line to <out> (TeacherContextBundle schema). Stop
after `--target N` accepted bundles.

Run inside polymarket-trader (needs pydantic + agents.research.rule_parser +
HTTPS_PROXY for Cloudflare-fronted endpoints):

    docker exec -i -e PYTHONPATH=/tmp/teacher_runner polymarket-trader \\
        python3 /tmp/teacher_runner/finetune/build_bundles.py \\
            --target 200 --out /tmp/teacher_runner/bundles.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


_GAMMA = "https://gamma-api.polymarket.com"
_MIN_VOLUME_USD = 1000.0
_MIN_NEWS = 5
_MICROSTRUCTURE_PATTERNS = [
    re.compile(r"\bUp or Down\b", re.I),
    re.compile(r"\d{1,2}:\d{2}(AM|PM)?-\d{1,2}:\d{2}", re.I),
    re.compile(
        r"\b(Set \d+|Set Handicap|Set Winner|Game Handicap|Match O/U|"
        r"Games O/U|O/U \d+\.\d+|Both Teams|First .* Goal|First To)\b",
        re.I,
    ),
    re.compile(r"\babove \d", re.I),
    re.compile(r"\bLoL:", re.I),
]


def _is_microstructure(question: str) -> bool:
    if not question:
        return True
    return any(p.search(question) for p in _MICROSTRUCTURE_PATTERNS)


def _is_substantive(question: str) -> bool:
    return bool(question) and len(question.split()) >= 8


def _proxy_url() -> Optional[str]:
    import os
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None


def _client(timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(proxy=_proxy_url(), timeout=timeout, follow_redirects=True)


def _gamma_page(end_min: datetime, end_max: datetime, offset: int, limit: int = 100) -> List[Dict[str, Any]]:
    params = {
        "closed": "true",
        "limit": limit,
        "offset": offset,
        "order": "endDate",
        "ascending": "false",
        "end_date_max": end_max.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date_min": end_min.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with _client(30.0) as c:
        r = c.get(f"{_GAMMA}/markets", params=params,
                  headers={"User-Agent": "phase4-build", "Accept": "application/json"})
        if r.status_code != 200:
            print(f"[gamma] HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return []
        return r.json()


def _gnews_count(query: str, start: datetime, end: datetime) -> int:
    """Count Google News RSS items for [start, end] window. Date-granular."""
    def fmt_d(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    q = f"{query} after:{fmt_d(start)} before:{fmt_d(end)}"
    with _client(20.0) as c:
        r = c.get(
            "https://news.google.com/rss/search",
            params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"User-Agent": "Mozilla/5.0 phase4-build"},
        )
        if r.status_code != 200:
            return -1
        return max(0, r.text.count("<item>"))


def _gnews_articles(query: str, start: datetime, end: datetime, max_n: int = 20) -> List[Dict[str, str]]:
    """Pull first max_n article {title, source} pairs for [start, end]."""
    def fmt_d(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    q = f"{query} after:{fmt_d(start)} before:{fmt_d(end)}"
    with _client(20.0) as c:
        r = c.get(
            "https://news.google.com/rss/search",
            params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"User-Agent": "Mozilla/5.0 phase4-build"},
        )
        if r.status_code != 200:
            return []
    # Cheap RSS parse without xml lib — title in <title>, source in <source url>
    items = []
    chunks = r.text.split("<item>")[1:max_n + 1]
    for chunk in chunks:
        title_m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", chunk, re.S)
        source_m = re.search(r"<source[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</source>", chunk, re.S)
        if title_m:
            items.append({
                "title": title_m.group(1).strip()[:300],
                "source": source_m.group(1).strip()[:80] if source_m else "",
            })
    return items


def _build_bundle(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a TeacherContextBundle dict from a Gamma market. Returns None
    if news count below threshold (partial-context filter)."""
    end = m.get("endDate")
    if not end:
        return None
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    decision_ts = end_dt - timedelta(hours=24)
    window_start = decision_ts - timedelta(days=7)
    query = " ".join((m.get("question") or "").split()[:6])

    news_count = _gnews_count(query, window_start, decision_ts)
    if news_count < _MIN_NEWS:
        return None
    articles = _gnews_articles(query, window_start, decision_ts, max_n=15)

    # Lazy import — only works inside polymarket-trader container.
    from agents.research.rule_parser import parse_market_rules  # type: ignore[import-not-found]
    parsed_rules = parse_market_rules(m)

    return {
        "bundle_version": 1,
        "market_id": m.get("conditionId") or str(m.get("id", "")),
        "decision_ts_utc": decision_ts.isoformat(),
        "market_payload": m,
        "parsed_rules": parsed_rules,
        "connector_summary_redacted": {
            "news": articles,
            "data_quality": {"unknown_count": 0, "news_count_total": news_count},
        },
        "intended_size_usd": 500,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200, help="bundles to produce")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-age-days", type=int, default=180)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    end_max = now
    end_min = now - timedelta(days=args.max_age_days)

    out_fh = open(args.out, "w")
    accepted = 0
    seen_cond: set[str] = set()
    page = 0
    page_limit = 100
    max_pages = 30
    rate_sleep_s = 1.2  # gnews is friendlier than gdelt; ~1 req/s sustained

    while accepted < args.target and page < max_pages:
        chunk = _gamma_page(end_min, end_max, offset=page * page_limit, limit=page_limit)
        if not chunk:
            break
        for m in chunk:
            if accepted >= args.target:
                break
            cond = m.get("conditionId")
            if not cond or cond in seen_cond:
                continue
            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = None
            if not (isinstance(outcomes, list) and len(outcomes) == 2):
                continue
            try:
                vol = float(m.get("volume") or 0.0)
            except (TypeError, ValueError):
                vol = 0.0
            if vol < _MIN_VOLUME_USD:
                continue
            q = m.get("question") or ""
            if _is_microstructure(q) or not _is_substantive(q):
                continue

            seen_cond.add(cond)
            try:
                bundle = _build_bundle(m)
            except Exception as exc:
                print(f"[build] err {cond[:10]}.. {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if bundle is None:
                continue  # news_count below threshold
            out_fh.write(json.dumps(bundle) + "\n")
            out_fh.flush()
            accepted += 1
            print(f"[{accepted:>3}/{args.target}] kept {cond[:10]}.. q={q[:60]!r} news={bundle['connector_summary_redacted']['data_quality']['news_count_total']}",
                  file=sys.stderr)
            time.sleep(rate_sleep_s)
        page += 1

    out_fh.close()
    print(f"\n[done] accepted={accepted}/{args.target} written to {args.out}", file=sys.stderr)
    return 0 if accepted >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
