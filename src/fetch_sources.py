"""
Freeze the real source text used to build the probe set.

Run once. Writes data/sources/snapshot.json with every record's origin URL,
identifier and retrieval time. build_probes.py reads only this snapshot, so a
later network fetch cannot change the released probe set.

Sources and licences:
  NWS active alerts   api.weather.gov        US federal work, public domain
  arXiv abstracts     export.arxiv.org/api   arXiv metadata, CC0 1.0
  AgentDojo attacks   github.com/ethz-spylab/agentdojo (MIT), templates copied
                      verbatim into build_probes.py with the commit hash
"""
import datetime
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sources" / "snapshot.json"
UA = {"User-Agent": "title36-probe-builder (research)", "Accept": "application/geo+json"}
ATOM = "{http://www.w3.org/2005/Atom}"

# Carrier and neutral-sentence pools are drawn from unrelated fields so that no
# carrier discusses prompt injection. The security query feeds only the
# attack-vocabulary hard negatives. The token query feeds the identifier
# hard negatives: benign sentences carrying a URL or code-like string.
ARXIV_QUERIES = {
    "carrier": [
        "cat:q-bio.PE", "cat:astro-ph.EP", "cat:econ.GN", "cat:physics.ao-ph",
        "cat:cs.DB", "cat:stat.AP", "cat:eess.SP", "cat:q-fin.GN",
    ],
    "vocab": [
        'abs:"ignore previous instructions"', 'abs:"ignore all previous instructions"',
        'abs:"prompt injection" AND abs:"instructions"',
    ],
    "token": ['abs:"code is available at"', 'abs:"github.com"'],
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def fetch_nws():
    url = "https://api.weather.gov/alerts/active?status=actual&message_type=alert"
    data = json.loads(get(url, UA))
    out, seen = [], set()
    for f in data.get("features", []):
        p = f["properties"]
        desc, instr = clean(p.get("description")), clean(p.get("instruction"))
        # Identical text is issued once per zone. Keep one copy.
        key = desc[:200]
        if not desc or key in seen:
            continue
        seen.add(key)
        out.append({
            "source": "nws", "id": p.get("id"), "event": p.get("event"),
            "sent": p.get("sent"), "url": f.get("id"),
            "description": desc, "instruction": instr,
        })
    return {"query_url": url, "retrieved": now(), "records": out}


def fetch_arxiv(query, n):
    q = urllib.parse.quote(query, safe=':')
    url = f"https://export.arxiv.org/api/query?search_query={q}&max_results={n}&sortBy=submittedDate&sortOrder=descending"
    root = ET.fromstring(get(url))
    recs = []
    for e in root.findall(f"{ATOM}entry"):
        recs.append({
            "source": "arxiv", "id": e.find(f"{ATOM}id").text.rsplit("/", 1)[-1],
            "title": clean(e.find(f"{ATOM}title").text),
            "abstract": clean(e.find(f"{ATOM}summary").text),
            "published": e.find(f"{ATOM}published").text, "query": query,
        })
    time.sleep(3.5)  # arXiv API asks for at least 3 s between calls
    return {"query_url": url, "retrieved": now(), "records": recs}


def main():
    snap = {"created": now(), "nws": fetch_nws(), "arxiv": {}}
    for pool, queries in ARXIV_QUERIES.items():
        snap["arxiv"][pool] = [fetch_arxiv(q, {'carrier': 40, 'vocab': 60, 'token': 60}[pool]) for q in queries]
    OUT.write_text(json.dumps(snap, indent=1, ensure_ascii=False))
    n_nws = len(snap["nws"]["records"])
    n_ax = {k: sum(len(b["records"]) for b in v) for k, v in snap["arxiv"].items()}
    print(f"nws alerts {n_nws}, arxiv {n_ax}, written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
