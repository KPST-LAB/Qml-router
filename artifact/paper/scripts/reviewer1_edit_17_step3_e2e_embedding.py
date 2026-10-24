#!/usr/bin/env python3
"""reviewer1_edit_17_step3_e2e_embedding.py

Purpose : reviewer1_edit_17 step3 - end-to-end accuracy with embedding-selected fact rows,
Record  : TEST-LEDGER 100 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_17_step3_e2e_embedding.py PORT EMB_PORT OUT_DIR [CONDS]   CONDS = lex,emb (default) | lex | emb
"""
import json, re, sqlite3, subprocess, sys, time, urllib.request
import numpy as np

PORT, EMB_PORT, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
CONDS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["lex", "emb"]
D = "$DATA"; Q = D + "/qml-router"; R = Q + "/ieee-rev1"
STORE = Q + "/store-lmes70/qml.db"
SUBSET = R + "/data/reviewer1_edit_17_step2_subset.db"
VEC = R + "/data/reviewer1_edit_17_step2_vectors_h200gpu"
DS = Q + "/longmemeval_s_cleaned.json"
K, FETCH, NPRED, EOT, RESERVE, N_CTX_SLOT = 32, 96, 96, 128009, 512, 16384
BLOCK_HEAD = "<|start_header_id|>user<|end_header_id|>\n\nKnown facts from my earlier conversations (with dates):\n"
BLOCK_TAIL = "<|eot_id|>"

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(port, path, payload, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

#------------------------------------------------------------
#Func : tok
#------------------------------------------------------------
def tok(text):
    return post(PORT, "/tokenize", {"content": text, "add_special": False})["tokens"]

#------------------------------------------------------------
#Func : norm
#------------------------------------------------------------
def norm(t):
    return re.sub(r"[^0-9a-z]+", "", t.lower())

#------------------------------------------------------------
#Func : f1
#------------------------------------------------------------
def f1(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

#------------------------------------------------------------
#Func : build_queries
#------------------------------------------------------------
def build_queries(texts):
    p = subprocess.run([D + "/t1_query"], input="\n".join(t.replace("\n", " ") for t in texts),
                       capture_output=True, text=True, timeout=300)
    return dict(l.split("\t", 1) for l in p.stdout.splitlines() if "\t" in l)

#------------------------------------------------------------
#Func : embed
#------------------------------------------------------------
def embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        for item in post(EMB_PORT, "/embedding", {"content": texts[i:i + bs]}, 600):
            v = item["embedding"][0] if isinstance(item["embedding"][0], list) else item["embedding"]
            out.append(v)
    a = np.asarray(out, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

ds = {it["question_id"]: it for it in json.load(open(DS))}
store = sqlite3.connect("file:%s?mode=ro" % STORE, uri=True, timeout=300)
sub = sqlite3.connect("file:%s?mode=ro" % SUBSET, uri=True)

inj = {}
for line in open(R + "/logs/reviewer1_edit_17_step0_srv_ask.log", errors="ignore"):
    m = re.search(r"session '([^']+)': facts mode, (\d+) summary line\(s\), (\d+) token", line)
    if m:
        inj[m.group(1)] = (int(m.group(2)), int(m.group(3)))
step0 = {}
for condition in ("msf", "trf"):
    for r in json.load(open(R + "/data/reviewer1_edit_17_step0_results_%s.json" % condition)):
        step0[r["qid"]] = r["answer"]

allv = np.load(VEC + ".npy").astype(np.float32); off = 0
text2vec = {}
for sid_v61, n in json.load(open(VEC + ".index.json")):
    rows = sub.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH 'tag:fact' AND session_id=?",
                       (sid_v61,)).fetchall()
    assert len(rows) == n, (sid_v61, len(rows), n)
    for (p, c), v in zip(rows, allv[off:off + n]):
        text2vec.setdefault((sid_v61, norm(c)), v)
    off += n

qids = [(condition, r["qid"]) for condition in ("msf", "trf")
        for r in json.load(open(R + "/data/reviewer1_edit_17_step0_results_%s.json" % condition))]
qtext = {qid: "Today is %s. %s" % (ds[qid].get("question_date", ""), ds[qid]["question"]) for _, qid in qids}
qmap = build_queries([ds[qid]["question"] for _, qid in qids])
qvec = {qid: v for (_, qid), v in zip(qids, embed([ds[qid]["question"] for _, qid in qids]))} if "emb" in CONDS else {}

#------------------------------------------------------------
#Func : live_tokens
#------------------------------------------------------------
def live_tokens(sid):
    t = [r[0] for r in store.execute("SELECT token_id FROM qml_token_log WHERE session_id=? ORDER BY seq_id", (sid,))]
    assert t and t[-1] == EOT, sid
    return t

#------------------------------------------------------------
#Func : wrapper_tokens
#------------------------------------------------------------
def wrapper_tokens(qid):
    return tok("<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" % qtext[qid])

#------------------------------------------------------------
#Func : select_lex
#------------------------------------------------------------
def select_lex(sid, qid):
    fq = qmap.get(ds[qid]["question"].replace("\n", " "), "")
    if not fq:
        return []
    got = store.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH ? AND session_id=? "
                        "ORDER BY rank LIMIT ?", ("tag:fact AND (%s)" % fq, sid, FETCH)).fetchall()
    seen, out = set(), []
    for p, c in got:
        k = norm(c)
        if k not in seen:
            seen.add(k); out.append(c)
            if len(out) == K:
                break
    return out

#------------------------------------------------------------
#Func : select_emb
#------------------------------------------------------------
def select_emb(sid, sid_v61, qid):
    rows = store.execute("SELECT turn_pos, content FROM qml_archive_fts WHERE qml_archive_fts MATCH 'tag:fact' AND session_id=?",
                         (sid,)).fetchall()
    texts, vecs = [], []
    for p, c in rows:
        v = text2vec.get((sid_v61, norm(c)))
        if v is not None:
            texts.append(c); vecs.append(v)
    if not texts:
        return []
    order = np.argsort(-(np.asarray(vecs) @ qvec[qid]), kind="stable")
    seen, out = set(), []
    for i in order:
        k = norm(texts[i])
        if k not in seen:
            seen.add(k); out.append(texts[i])
            if len(out) == K:
                break
    return out

#------------------------------------------------------------
#Func : run_arm
#------------------------------------------------------------
def run_arm(name, selector):
    results = {"msf": [], "trf": []}; agree = tot = 0; inj_match = 0
    t0 = time.time()
    for i, (condition, qid) in enumerate(qids):
        sid = "lme-r1c17s0%s::%s" % (condition, qid); sid_v61 = "lme-%s::%s" % (condition, qid)
        live = live_tokens(sid); wrap = wrapper_tokens(qid)
        rows = selector(sid, sid_v61, qid) if selector is select_emb else selector(sid, qid)
        block = tok(BLOCK_HEAD + "".join("- %s\n" % c for c in rows) + BLOCK_TAIL) if rows else []
        avail = max(0, N_CTX_SLOT - len(live) - len(wrap) - RESERVE)
        block = block[:avail]
        prompt = live + block + wrap
        d = post(PORT, "/completion", {"prompt": prompt, "n_predict": NPRED, "temperature": 0, "cache_prompt": False})
        ans = (d.get("content") or "").strip()
        gold = str(ds[qid]["answer"])
        sub_ok = gold.lower() in ans.lower(); sc = f1(ans, gold)
        rec = {"qid": qid, "type": ds[qid]["question_type"], "ok": bool(sub_ok or sc >= 0.5), "sub": bool(sub_ok),
               "f1": round(sc, 3), "answer": ans[:4000], "gold": gold,
               "n_rows": len(rows), "n_block_tokens": len(block), "engine_step0_rows_tokens": inj.get(sid)}
        results[condition].append(rec)
        tot += 1
        if name == "lex":
            agree += (ans == (step0.get(qid) or "").strip())
            inj_match += (inj.get(sid) == (len(rows), len(block)))
        if (i + 1) % 25 == 0:
            print("[%s] %d/%d, %.0fs" % (name, i + 1, len(qids), time.time() - t0), flush=True)
    for condition in ("msf", "trf"):
        json.dump(results[condition], open("%s/results_step3_%s_%s.json" % (OUT, name, condition), "w"), indent=1)
    if name == "lex":
        print("[lex] reproduction: answers identical to step0 %d/%d; injected (rows, tokens) identical to the engine log %d/%d"
              % (agree, tot, inj_match, tot), flush=True)
    for condition in ("msf", "trf"):
        n = len(results[condition]); ok = sum(r["ok"] for r in results[condition])
        print("[%s] %s string-match metric %d/%d = %.3f (official judge follows)" % (name, condition, ok, n, ok / n), flush=True)

if "lex" in CONDS:
    run_arm("lex", select_lex)
if "emb" in CONDS:
    run_arm("emb", select_emb)
print("done", flush=True)
