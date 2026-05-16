"""Phase 3 reconstruction check — runs against the live polymarket-trader.

Sampling strategy:
- 20 resolved Polymarket markets, sourced from Gamma API closed=true
- Bucketed: 7 markets resolved 0-30d ago (recent),
            7 markets 30-90d (mid),
            6 markets 90-180d (old)
- Markets ordered by `endDate desc`, filtered to YES/NO binary with sufficient volume

Per market reconstruction probes (executed inside polymarket-trader for
network + connector availability):
- Trade history at decision moment = endDate - 24h:
    orderFilledEvents in [decision-24h, decision]; count > 0 → trades_ok
- News window covering [decision-7d, decision]:
    GDELT DOC API with STARTDATETIME/ENDDATETIME absolute params;
    count >= 5 → news_ok

Classification:
    full          — trades_ok AND news_ok
    degraded      — exactly one of the two
    unreconstructable — neither (or fetch errored)

Gate: ≥ 70% of the 20 markets classified as `full` within the 180d window.

Run:
    python3 -m finetune.data_archival.check_reconstruction --out report.json

Costs: GDELT + Polymarket subgraph free. ~0 dollar spend.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import urllib.request
import urllib.error
import urllib.parse


# -----------------------------------------------------------------------------
# Gamma API — closed-market sampling
# -----------------------------------------------------------------------------

_GAMMA = "https://gamma-api.polymarket.com"


def _http_get_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 phase3-recon",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


_MIN_VOLUME_USD = 1000.0  # filters obvious spam; mid/old markets often <10k

# Drop high-frequency microstructure questions that resolve every few minutes
# and have no meaningful news signal. Pattern-matched on the question text.
import re as _re
_MICROSTRUCTURE_PATTERNS = [
    _re.compile(r"\bUp or Down\b", _re.I),
    _re.compile(r"\d{1,2}:\d{2}(AM|PM)?-\d{1,2}:\d{2}", _re.I),  # 5-min window
    _re.compile(r"\b(Set \d+|Set Handicap|Set Winner|Game Handicap|Match O/U|Games O/U|O/U \d+\.\d+|Both Teams|First .* Goal|First To)\b", _re.I),
    _re.compile(r"\babove \d", _re.I),  # "Ethereum above 2,405 on …"
    _re.compile(r"\bLoL:", _re.I),
]


def _is_substantive(question: str) -> bool:
    """Substantive markets have ≥ 8 words and an actual subject (not just team-vs-team)."""
    if not question:
        return False
    words = question.split()
    if len(words) < 8:
        return False
    return True


def _is_microstructure(question: str) -> bool:
    if not question:
        return True
    for p in _MICROSTRUCTURE_PATTERNS:
        if p.search(question):
            return True
    return False


def _query_bucket(now_utc: datetime, days_min: int, days_max: int, quota: int) -> List[Dict[str, Any]]:
    """Fetch closed markets resolved between (now-days_max, now-days_min).
    Paginates through Gamma (server caps each page at 100)."""
    end_max = (now_utc - timedelta(days=days_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_min = (now_utc - timedelta(days=days_max)).strftime("%Y-%m-%dT%H:%M:%SZ")
    page_limit = 100
    max_pages = 10  # 1000 markets ought to be enough for any bucket
    raw: List[Dict[str, Any]] = []
    for page in range(max_pages):
        offset = page * page_limit
        url = (
            f"{_GAMMA}/markets?closed=true&limit={page_limit}&offset={offset}"
            f"&order=endDate&ascending=false&end_date_max={end_max}&end_date_min={end_min}"
        )
        chunk = _http_get_json(url)
        if not chunk:
            break
        raw.extend(chunk)
        if len(chunk) < page_limit:
            break
    out: List[Dict[str, Any]] = []
    for m in raw:
        end = m.get("endDate")
        if not end:
            continue
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            continue
        age_days = (now_utc - end_dt).days
        # Binary YES/NO only.
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                outcomes = None
        if not (isinstance(outcomes, list) and len(outcomes) == 2):
            continue
        # Volume filter — drop high-frequency 5-min crypto / sports spam.
        try:
            vol = float(m.get("volume") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol < _MIN_VOLUME_USD:
            continue
        q = m.get("question") or ""
        if _is_microstructure(q):
            continue
        if not _is_substantive(q):
            continue
        out.append({"market": m, "age_days": age_days})
        if len(out) >= quota:
            break
    return out


def _sample_markets(now_utc: datetime) -> List[Dict[str, Any]]:
    """Bucketed pull: 7 recent (0-30d), 7 mid (30-90d), 6 old (90-180d)."""
    sample: List[Dict[str, Any]] = []
    for bucket_name, (lo, hi, q) in (
        ("recent", (0, 30, 7)),
        ("mid",    (30, 90, 7)),
        ("old",    (90, 180, 6)),
    ):
        rows = _query_bucket(now_utc, lo, hi, q)
        for r in rows:
            r["bucket"] = bucket_name
        sample.extend(rows)
    return sample


# -----------------------------------------------------------------------------
# Probe execution inside polymarket-trader
# -----------------------------------------------------------------------------

# Inner script — runs inside polymarket-trader to inherit its httpx config
# (HTTPS_PROXY=docker.home.network:3128, which fronts Cloudflare-protected
# endpoints). Uses the polymarket-agents shared http_client when possible,
# falls back to httpx direct for endpoints not wrapped there.
_PROBE_SCRIPT = r'''
import json, sys, traceback
from datetime import datetime, timedelta, timezone

import httpx
from agents.utils.http_client import get_proxy_url

req = json.loads(sys.stdin.read())
cond_id = req["condition_id"]
decision_ts = datetime.fromisoformat(req["decision_ts"])
question = req["question"]
window_start = decision_ts - timedelta(days=7)
trade_window_start = decision_ts - timedelta(hours=24)

errors = []
proxy = get_proxy_url()

def _client(timeout=20.0):
    # httpx accepts a single proxy URL for both http/https schemes.
    return httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True)

# --- Trade history via Polymarket data-api /trades ---
trades_count = -1
try:
    with _client(15.0) as c:
        # Try data-api first (better signal: actual price-tick trades for THIS market)
        r = c.get(
            "https://data-api.polymarket.com/trades",
            params={
                "market": cond_id,
                "limit": 100,
                "startTs": int(trade_window_start.timestamp()),
                "endTs":   int(decision_ts.timestamp()),
            },
        )
        if r.status_code == 200:
            d = r.json()
            if isinstance(d, list):
                trades_count = len(d)
            else:
                errors.append(f"data-api: unexpected shape {type(d).__name__}")
        else:
            errors.append(f"data-api: HTTP {r.status_code}: {r.text[:120]}")
except Exception as exc:
    errors.append(f"data-api: {type(exc).__name__}: {exc}")

# --- News window: GDELT primary, Google News RSS fallback ---
news_count = -1
news_source = None
try:
    def fmt(dt):
        return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    query = " ".join(question.split()[:6])
    with _client(20.0) as c:
        r = c.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": "75",
                "startdatetime": fmt(window_start),
                "enddatetime":   fmt(decision_ts),
            },
            headers={"User-Agent": "phase3-recon"},
        )
        if r.status_code == 200:
            try:
                gdoc = r.json()
                news_count = len(gdoc.get("articles", []))
                news_source = "gdelt"
            except json.JSONDecodeError:
                body = r.text
                if "rate" in body.lower():
                    errors.append("gdelt: rate-limited (falling back to gnews)")
                else:
                    errors.append(f"gdelt: non-json body len={len(body)}")
        elif r.status_code == 429:
            errors.append("gdelt: rate-limited (falling back to gnews)")
        else:
            errors.append(f"gdelt: HTTP {r.status_code}")
except Exception as exc:
    errors.append(f"gdelt: {type(exc).__name__}: {exc}")

if news_count < 0:
    # Google News RSS fallback. Date filter is day-granular (after:/before:).
    # Returns RSS XML; we count <item> tags.
    try:
        def fmt_d(dt):
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        gnews_q = (
            f"{query} after:{fmt_d(window_start)} before:{fmt_d(decision_ts)}"
        )
        with _client(20.0) as c:
            r = c.get(
                "https://news.google.com/rss/search",
                params={"q": gnews_q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "Mozilla/5.0 phase3-recon"},
            )
            if r.status_code == 200:
                # Cheap count: split on "<item>" — RSS items are well-formed,
                # avoid pulling in xml.etree for a row count.
                news_count = max(0, r.text.count("<item>"))
                news_source = "gnews"
            else:
                errors.append(f"gnews: HTTP {r.status_code}")
    except Exception as exc:
        errors.append(f"gnews: {type(exc).__name__}: {exc}")

print(json.dumps({
    "trades_count": trades_count,
    "news_count": news_count,
    "news_source": news_source,
    "errors": errors,
}))
'''


def _probe_market(market: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke _PROBE_SCRIPT inside polymarket-trader for one market."""
    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    decision = end - timedelta(hours=24)
    req_payload = json.dumps({
        "condition_id": market.get("conditionId", ""),
        "decision_ts": decision.isoformat(),
        "question": market.get("question", ""),
    })
    cmd = [
        "docker", "exec", "-i", "polymarket-trader",
        "python3", "-c", _PROBE_SCRIPT,
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=req_payload, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"trades_count": -1, "news_count": -1, "errors": ["subprocess timeout"], "elapsed_s": 60.0}
    if proc.returncode != 0:
        return {"trades_count": -1, "news_count": -1,
                "errors": [f"rc={proc.returncode} stderr={proc.stderr[-200:]}"],
                "elapsed_s": time.monotonic() - t0}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"trades_count": -1, "news_count": -1,
                "errors": [f"non-json stdout={proc.stdout[:200]!r}"],
                "elapsed_s": time.monotonic() - t0}
    result["elapsed_s"] = time.monotonic() - t0
    return result


def _classify(trades: int, news: int) -> str:
    trades_ok = trades > 0
    news_ok = news >= 5
    if trades_ok and news_ok:
        return "full"
    if trades_ok or news_ok:
        return "degraded"
    return "unreconstructable"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("finetune/REVIEWS/phase3_reconstruction.json"))
    ap.add_argument("--max-markets", type=int, default=20)
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"[gamma] sampling closed markets…", file=sys.stderr)
    samples = _sample_markets(now_utc)
    samples = samples[: args.max_markets]
    print(f"[gamma] sampled {len(samples)} markets", file=sys.stderr)

    # GDELT enforces ~1 req per 5s per IP. The probe inside each market hits
    # GDELT once, so we space markets by GDELT_DELAY_S.
    GDELT_DELAY_S = 5.5

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        if i > 0:
            time.sleep(GDELT_DELAY_S)
        m = s["market"]
        print(
            f"[{i + 1:>2}/{len(samples)}] bucket={s['bucket']:6} age={s['age_days']:>4}d "
            f"cond={(m.get('conditionId') or '')[:10]}.. q={(m.get('question') or '')[:50]!r}",
            file=sys.stderr,
        )
        probe = _probe_market(m)
        cls = _classify(probe["trades_count"], probe["news_count"])
        results.append({
            "condition_id": m.get("conditionId"),
            "question":     m.get("question"),
            "end_date":     m.get("endDate"),
            "bucket":       s["bucket"],
            "age_days":     s["age_days"],
            "classification": cls,
            "trades_count": probe["trades_count"],
            "news_count":   probe["news_count"],
            "news_source":  probe.get("news_source"),
            "elapsed_s":    round(probe.get("elapsed_s", 0.0), 1),
            "errors":       probe.get("errors", []),
        })
        print(f"     -> {cls:18} trades={probe['trades_count']} news={probe['news_count']}"
              f"({probe.get('news_source')}) in {probe.get('elapsed_s', 0):.1f}s",
              file=sys.stderr)

    # Tally
    total = len(results)
    rates = {cls: sum(1 for r in results if r["classification"] == cls) / max(total, 1)
             for cls in ("full", "degraded", "unreconstructable")}
    report = {
        "generated_at_utc": now_utc.isoformat(),
        "n_markets": total,
        "rates": rates,
        "gate_pass": rates["full"] >= 0.70,
        "markets": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {args.out}", file=sys.stderr)
    print(f"[done] full={rates['full']:.0%} degraded={rates['degraded']:.0%} "
          f"unrec={rates['unreconstructable']:.0%}  gate {'PASS' if report['gate_pass'] else 'FAIL'}",
          file=sys.stderr)
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
