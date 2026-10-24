#!/usr/bin/env python3
"""reviewer1_edit_17_step1_retrieval_curve.py

Purpose : reviewer1_edit_17 step1 - reader-free retrieval comparison for reviewer 1, comment 17
Record  : TEST-LEDGER 98 (docs/3-status/TEST-LEDGER.md)
"""
import json, re, sqlite3, subprocess, sys, time, urllib.request
import numpy as np

D = "$DATA"; Q = D + "/qml-router"; OUT = Q + "/ieee-rev1/data"
EMB = "http://127.0.0.1:58172"
KS = [3, 5, 8, 12, 24, 32, 50]
LEX_LIMIT = 150

con = sqlite3.connect(f"file:{Q}/store-lmes70/qml.db?mode=ro", uri=True, timeout=120)
data = {it["question_id"]: it for it in json.load(open(Q + "/longmemeval_s_cleaned.json"))}

#------------------------------------------------------------
#Func : norm
#------------------------------------------------------------
def norm(text):
    return re.sub(r"[^0-9a-z]+", "", text.lower())

#------------------------------------------------------------
#Func : build_queries
#------------------------------------------------------------
def build_queries(questions):
    r = subprocess.run([D + "/t1_query"], input="\n".join(q.replace("\n", " ") for q in questions),
                       capture_output=True, text=True, timeout=180)
    return dict(l.split("\t", 1) for l in r.stdout.splitlines() if "\t" in l)

#------------------------------------------------------------
#Func : embed
#------------------------------------------------------------
def embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        req = urllib.request.Request(EMB + "/embedding", data=json.dumps({"content": texts[i:i + bs]}).encode(),
                                     headers={"Content-Type": "application/json"})
        for item in json.loads(urllib.request.urlopen(req, timeout=600).read()):
            v = item["embedding"][0] if isinstance(item["embedding"][0], list) else item["embedding"]
            out.append(v)
    a = np.asarray(out, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

#------------------------------------------------------------
#Func : dedup_positions
#------------------------------------------------------------
def dedup_positions(ranked):
    seen, pos = set(), []
    for p, c in ranked:
        k = norm(c)
        if k not in seen:
            seen.add(k); pos.append(p)
    return pos

#------------------------------------------------------------
#Func : score
#------------------------------------------------------------
def score(pos, ranges, counts_any, counts_all):
    for k in KS:
        top = pos[:k]
        counts_any[k] += any(a <= p < b for a, b in ranges for p in top)
        counts_all[k] += all(any(a <= p < b for p in top) for a, b in ranges)

result = {}
t_start = time.time()
for condition, rfile in (("msf", "results_msf_final_paperjudged.json"), ("trf", "results_trf_final_paperjudged.json")):
    rows = json.load(open(f"{Q}/{rfile}"))
    side = json.load(open(f"{Q}/answers_{condition}.json"))
    qmap = build_queries([data[r["qid"]]["question"] for r in rows])
    cnt = {m: ({k: 0 for k in KS}, {k: 0 for k in KS}) for m in ("lexical", "embedding")}
    n = skipped = n_fact_rows = 0
    per_q = []
    t_arm = time.time()
    for idx, r in enumerate(rows):
        qid = r["qid"]; n += 1
        ranges = side.get(qid) or []
        question = data[qid]["question"]
        mq = qmap.get(question.replace("\n", " "), "")
        sid = f"lme-{condition}::{qid}"
        if not ranges:
            skipped += 1
            continue
        lex = con.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH ? AND session_id=? "
                          "ORDER BY rank LIMIT ?", (f"tag:fact AND ({mq})", sid, LEX_LIMIT)).fetchall() if mq else []
        lex_pos = dedup_positions(lex)
        facts = con.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH ? AND session_id=?",
                            ("tag:fact", sid)).fetchall()
        n_fact_rows += len(facts)
        if facts:
            fv = embed([c for _, c in facts]); qv = embed([question])[0]
            order = np.argsort(-(fv @ qv), kind="stable")
            emb_pos = dedup_positions([facts[i] for i in order])
        else:
            emb_pos = []
        score(lex_pos, ranges, *cnt["lexical"])
        score(emb_pos, ranges, *cnt["embedding"])
        per_q.append({"qid": qid, "n_fact_rows": len(facts), "n_ranges": len(ranges),
                      "lexical_top50": lex_pos[:50], "embedding_top50": emb_pos[:50]})
        if (idx + 1) % 25 == 0:
            print(f"[{condition}] {idx + 1}/{len(rows)} questions, {n_fact_rows} fact rows embedded, "
                  f"{time.time() - t_arm:.0f}s", flush=True)
    result[condition] = {"n": n, "skipped_no_evidence": skipped, "fact_rows": n_fact_rows,
                   **{m: {"any": cnt[m][0], "all": cnt[m][1]} for m in cnt}}
    json.dump(per_q, open(f"{OUT}/reviewer1_edit_17_step1_retrieval_perquestion_{condition}.json", "w"))

ref = json.load(open(D + "/kcurve133.json"))
ref32 = {"msf": 116, "trf": 119}
print("--- checks (lexical ranker must reproduce the v61 Fig. 6(a) record) ---")
for condition in ("msf", "trf"):
    for k in (3, 5, 8, 12, 24, 50):
        for m in ("any", "all"):
            a, b = result[condition]["lexical"][m][k], ref[condition][m][str(k)]
            print(f"{condition} {m}@{k}: s1 {a} / kcurve133 {b} -> {'OK' if a == b else 'MISMATCH'}")
    a = result[condition]["lexical"]["all"][32]
    print(f"{condition} all@32: s1 {a} / ledger {ref32[condition]} -> {'OK' if a == ref32[condition] else 'MISMATCH'}")

print("--- recall@k, lexical vs embedding (count / n) ---")
for condition in ("msf", "trf"):
    n = result[condition]["n"]
    for m in ("any", "all"):
        for ranker in ("lexical", "embedding"):
            vals = " ".join(f"@{k}={result[condition][ranker][m][k]}" for k in KS)
            print(f"{condition} recall_{m} {ranker:9s} (n={n}): {vals}")
json.dump(result, open(f"{OUT}/reviewer1_edit_17_step1_retrieval_curve.json", "w"), indent=1)
print(f"saved {OUT}/reviewer1_edit_17_step1_retrieval_curve.json, total {time.time() - t_start:.0f}s")
