"""
Freeze the external corpora for the detector panel (phase 11).

  NotInject    leolee99/NotInject (MIT), the over-defense benchmark of
               InjecGuard/PIGuard: benign prompts that carry one, two or three
               trigger words. All three splits, every item benign.
  deepset      deepset/prompt-injections (Apache-2.0), test split, the
               publisher's labels (1 = injection).
  title35      the probe set of the companion study, read from its folder.

Writes data/thirdparty/panel_external.jsonl with the source revision of every
item, so the panel never fetches at run time.
"""
import json
import pathlib
import random
import re

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "thirdparty" / "panel_external.jsonl"
NOTINJECT = ("leolee99/NotInject", "847ae76cf8fea5ed325429e569ae8cfef022d2e0")
DEEPSET = "deepset/prompt-injections"
TITLE35 = ROOT.parent / "title35-prompt-signal-weight-calibration" / "data" / "probes" / "probes.jsonl"
PROMPTS = ("fka/prompts.chat", "fbea17f2045d053d27f1de9f099e9bfdbe55bf47")
# Community prompt collections include jailbreak prompts, which are not
# benign. A prompt carrying any of these markers is excluded before sampling.
JAILBREAK_MARKERS = re.compile(r"(?i)\bDAN\b|developer mode|jailbreak|no restrictions|without restrictions|"
                               r"ignore (all |any )?(previous|prior)|uncensored|unfiltered|no ethical|"
                               r"do anything now|bypass|no limits|evil|amoral")
N_PROMPTS = 200


def main():
    rows = []
    for split in ("one", "two", "three"):
        f = hf_hub_download(NOTINJECT[0], f"data/NotInject_{split}-00000-of-00001.parquet",
                            repo_type="dataset", revision=NOTINJECT[1])
        for i, r in pd.read_parquet(f).iterrows():
            rows.append({"id": f"notinject_{split}_{i:03d}", "corpus": "notinject", "text": r["prompt"],
                         "label": "benign", "group": f"notinject_{split}",
                         "trigger_words": list(r["word_list"]), "source": f"{NOTINJECT[0]}@{NOTINJECT[1][:12]}"})
    ds_rev = HfApi().dataset_info(DEEPSET).sha
    files = [s.rfilename for s in HfApi().dataset_info(DEEPSET, revision=ds_rev).siblings if "test" in s.rfilename]
    d = pd.read_parquet(hf_hub_download(DEEPSET, files[0], repo_type="dataset", revision=ds_rev))
    for i, r in d.iterrows():
        rows.append({"id": f"deepset_{i:03d}", "corpus": "deepset", "text": r["text"],
                     "label": "attack" if int(r["label"]) == 1 else "benign",
                     "group": "deepset", "source": f"{DEEPSET}@{ds_rev[:12]}"})
    for l in TITLE35.read_text().splitlines():
        p = json.loads(l)
        rows.append({"id": f"t35_{p['id']}", "corpus": "title35", "text": p["text"],
                     "label": "benign" if p["true_label"] == "SAFE" else "attack",
                     "group": f"t35_{p['phrasing']}", "is_prompt": p["is_prompt"], "source": "title35 probe set"})
    # Real instruction-shaped benign text: community role prompts (CC0).
    pc = pd.read_csv(hf_hub_download(PROMPTS[0], "prompts.csv", repo_type="dataset", revision=PROMPTS[1]))
    pool = [(i, r["act"], r["prompt"]) for i, r in pc.iterrows()
            if isinstance(r["prompt"], str) and not JAILBREAK_MARKERS.search(r["prompt"] + " " + str(r["act"]))]
    n_excluded = len(pc) - len(pool)
    random.Random(20260925).shuffle(pool)
    for i, act, prompt in sorted(pool[:N_PROMPTS]):
        rows.append({"id": f"prompts_{i:04d}", "corpus": "prompts_chat", "text": prompt, "label": "benign",
                     "group": "role_prompt", "act": act, "source": f"{PROMPTS[0]}@{PROMPTS[1][:12]}"})
    # Real security documentation: arXiv abstracts about prompt injection
    # (arXiv metadata, CC0), from the companion study's frozen snapshot.
    snap = json.loads((ROOT / "data" / "sources" / "snapshot.json").read_text())
    seen = set()
    for b in snap["arxiv"]["vocab"]:
        for r in b["records"]:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append({"id": f"arxivsec_{r['id']}", "corpus": "arxiv_security", "text": r["abstract"],
                         "label": "benign", "group": "security_abstract", "source": f"arXiv:{r['id']}"})
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (OUT.parent / "PANEL_PROVENANCE.json").write_text(json.dumps(
        {"notinject": {"repo": NOTINJECT[0], "revision": NOTINJECT[1], "licence": "MIT"},
         "deepset": {"repo": DEEPSET, "revision": ds_rev, "file": files[0], "licence": "Apache-2.0"},
         "prompts_chat": {"repo": PROMPTS[0], "revision": PROMPTS[1], "licence": "CC0-1.0",
                          "excluded_by_jailbreak_markers": n_excluded, "sampled": N_PROMPTS, "seed": 20260925},
         "arxiv_security": {"source": "data/sources/snapshot.json, arXiv vocab queries", "licence": "CC0-1.0"},
         "counts": {c: sum(r["corpus"] == c for r in rows)
                    for c in ("notinject", "deepset", "title35", "prompts_chat", "arxiv_security")}}, indent=1))
    print({c: sum(r["corpus"] == c for r in rows) for c in ("notinject", "deepset", "title35", "prompts_chat", "arxiv_security")}, "excluded", n_excluded)


if __name__ == "__main__":
    main()
