#!/usr/bin/env python3
"""reviewer1_edit_17_step2_retrieval_latency.py

Purpose : reviewer1_edit_17 step2, part 2 - retrieval latency, lexical versus embedding (reviewer 1, comment 17;
Record  : TEST-LEDGER 99 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_17_step2_retrieval_latency.py DB QUESTIONS EMB_URL OUT_PREFIX WRITE_MODE [VEC_CACHE]
"""
import json, re, sqlite3, statistics, sys, time, urllib.request
import numpy as np

DB, QFILE, EMB, OUT, MODE = sys.argv[1:6]
VEC_CACHE = sys.argv[6] if len(sys.argv) > 6 else OUT + "_vectors"
K, LEX_LIMIT, R, BS = 32, 150, 5, 64
now = time.perf_counter

#------------------------------------------------------------
#Func : norm
#------------------------------------------------------------
def norm(t):
    return re.sub(r"[^0-9a-z]+", "", t.lower())

#------------------------------------------------------------
#Func : dedup_top
#------------------------------------------------------------
def dedup_top(ranked, k):
    seen, pos = set(), []
    for p, c in ranked:
        key = norm(c)
        if key not in seen:
            seen.add(key); pos.append(p)
            if len(pos) == k:
                break
    return pos

#------------------------------------------------------------
#Func : embed
#------------------------------------------------------------
def embed(texts):
    out = []
    for i in range(0, len(texts), BS):
        req = urllib.request.Request(EMB + "/embedding", data=json.dumps({"content": texts[i:i + BS]}).encode(),
                                     headers={"Content-Type": "application/json"})
        for item in json.loads(urllib.request.urlopen(req, timeout=600).read()):
            v = item["embedding"][0] if isinstance(item["embedding"][0], list) else item["embedding"]
            out.append(v)
    a = np.asarray(out, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

#------------------------------------------------------------
#Func : recall_all
#------------------------------------------------------------
def recall_all(pos, ranges):
    return all(any(a <= p < b for p in pos) for a, b in ranges)

#------------------------------------------------------------
#Func : stats
#------------------------------------------------------------
def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "median_ms": round(1e3 * statistics.median(xs), 3),
            "p95_ms": round(1e3 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 3),
            "mean_ms": round(1e3 * sum(xs) / len(xs), 3), "total_s": round(sum(xs), 2)}

con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
qs = json.load(open(QFILE))
print("device run: db=%s questions=%d emb=%s mode=%s sqlite=%s" % (DB, len(qs), EMB, MODE, sqlite3.sqlite_version), flush=True)

facts, vecs, write_fetch, write_embed = {}, {}, [], []
sample = int(MODE.split(":")[1]) if MODE.startswith("sample:") else None
for i, q in enumerate(qs):
    sid = q["session_id"]
    t = now()
    rows = con.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH 'tag:fact' "
                       "AND session_id=?", (sid,)).fetchall()
    tf = now() - t
    facts[sid] = rows
    if MODE == "all" or (sample is not None and i < sample):
        t = now(); v = embed([c for _, c in rows]); te = now() - t
        write_fetch.append(tf); write_embed.append(te)
        if MODE == "all":
            vecs[sid] = v
        if (i + 1) % 25 == 0 or (sample is not None and i + 1 == sample):
            print("[write] %d sessions, last %d rows fetch %.1f ms embed %.0f ms" % (i + 1, len(rows), 1e3 * tf, 1e3 * te), flush=True)
if MODE == "all":
    sids = [q["session_id"] for q in qs]
    np.save(VEC_CACHE + ".npy", np.concatenate([vecs[s] for s in sids]).astype(np.float16))
    json.dump([[s, len(vecs[s])] for s in sids], open(VEC_CACHE + ".index.json", "w"))
else:
    allv = np.load(VEC_CACHE + ".npy").astype(np.float32); off = 0
    for s, n in json.load(open(VEC_CACHE + ".index.json")):
        vecs[s] = allv[off:off + n]; off += n
        assert len(vecs[s]) == len(facts[s]), "vector cache does not match the fact rows of " + s

per_q, lex_cold, lex_warm, eq_cold, eq_warm, er_warm = [], [], [], [], [], []
rec = {"lexical": 0, "embedding": 0}
for q in qs:
    sid, fq = q["session_id"], q["fts_query"]
    lt = []
    for r in range(R):
        t = now()
        got = con.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH ? AND session_id=? "
                          "ORDER BY rank LIMIT ?", ("tag:fact AND (%s)" % fq, sid, LEX_LIMIT)).fetchall() if fq else []
        lpos = dedup_top(got, K)
        lt.append(now() - t)
    et, rt = [], []
    for r in range(R):
        t = now(); qv = embed([q["question"]])[0]; t1 = now()
        order = np.argsort(-(vecs[sid] @ qv), kind="stable")
        epos = dedup_top([facts[sid][j] for j in order], K)
        t2 = now()
        et.append(t1 - t); rt.append(t2 - t1)
    lex_cold.append(lt[0]); lex_warm.append(statistics.median(lt[1:]))
    eq_cold.append(et[0] + rt[0]); eq_warm.append(statistics.median([a + b for a, b in zip(et[1:], rt[1:])]))
    er_warm.append(statistics.median(rt[1:]))
    if q["ranges"]:
        rec["lexical"] += recall_all(lpos, q["ranges"]); rec["embedding"] += recall_all(epos, q["ranges"])
    per_q.append({"condition": q["condition"], "qid": q["qid"], "n_fact_rows": len(facts[sid]),
                  "lexical_ms": [round(1e3 * x, 3) for x in lt],
                  "embed_question_ms": [round(1e3 * x, 3) for x in et], "rank_ms": [round(1e3 * x, 3) for x in rt]})

summary = {
    "lexical_query_cold": stats(lex_cold), "lexical_query_warm": stats(lex_warm),
    "embedding_query_cold": stats(eq_cold), "embedding_query_warm": stats(eq_warm),
    "embedding_rank_only_warm": stats(er_warm),
    "embedding_write_fetch_per_session": stats(write_fetch), "embedding_write_embed_per_session": stats(write_embed),
    "embedding_write_rows": sum(len(facts[q["session_id"]]) for i, q in enumerate(qs)
                                if MODE == "all" or (sample is not None and i < sample)),
    "recall_all_at_32": rec, "mode": MODE, "sqlite": sqlite3.sqlite_version,
}
if write_embed:
    summary["embedding_write_rows_per_s"] = round(summary["embedding_write_rows"] / sum(write_embed), 1)
json.dump({"summary": summary, "per_question": per_q}, open(OUT + ".json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
