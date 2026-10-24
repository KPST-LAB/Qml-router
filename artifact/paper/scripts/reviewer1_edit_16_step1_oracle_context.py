#!/usr/bin/env python3
"""reviewer1_edit_16_step1_oracle_context.py

Purpose : reviewer1_edit_16 step1 - oracle-context reading
Record  : TEST-LEDGER 101 (docs/3-status/TEST-LEDGER.md)
Usage   : reviewer1_edit_16_step1_oracle_context.py PORT DATASET_JSON REF_MSF_JSON REF_TRF_JSON OUT_JSON [N_CTX]
"""
import json, sys, time, urllib.request

PORT, DATASET, REF_MSF, REF_TRF, OUT = sys.argv[1:6]
N_CTX = int(sys.argv[6]) if len(sys.argv) > 6 else 32768
N_PREDICT = 256
BASE = "http://127.0.0.1:%s" % PORT
SYS_TXT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           "You are a helpful assistant. Answer briefly and directly.<|eot_id|>")

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : ntok
#------------------------------------------------------------
def ntok(text):
    return len(post("/tokenize", {"content": text}, timeout=600)["tokens"])

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
#Func : session_text
#------------------------------------------------------------
def session_text(idx, date, turns):
    lines = ["### Session %d (date: %s)" % (idx, date)]
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append("%s: %s" % (role, t.get("content", "")))
    return "\n".join(lines)

#------------------------------------------------------------
#Func : build_prompt
#Same reading prompt as the full-history baseline, over the sessions in keep_idx
#------------------------------------------------------------
def build_prompt(item, keep_idx):
    sess = [item["haystack_sessions"][i] for i in keep_idx]
    dates = [item["haystack_dates"][i] for i in keep_idx]
    hist = "\n\n".join(session_text(n + 1, d, s) for n, (d, s) in enumerate(zip(dates, sess)))
    body = ("I will give you several history chats between you and a user. Please answer "
            "the question based on the relevant chat history.\n\nHistory Chats:\n\n"
            "%s\n\nCurrent Date: %s\nQuestion: %s\nAnswer:" % (hist, item["question_date"], item["question"]))
    return (SYS_TXT + "<|start_header_id|>user<|end_header_id|>\n\n" + body +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

ds = {it["question_id"]: it for it in json.load(open(DATASET))}
targets = [r["qid"] for r in json.load(open(REF_MSF))] + [r["qid"] for r in json.load(open(REF_TRF))]
rows = []
t0 = time.time()
for i, qid in enumerate(targets):
    item = ds[qid]
    evid = set(item.get("answer_session_ids", []))
    keep_idx = [k for k, s in enumerate(item["haystack_session_ids"]) if s in evid]
    prompt = build_prompt(item, keep_idx)
    n = ntok(prompt)
    truncated = n + N_PREDICT > N_CTX
    d = post("/completion", {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False})
    ans = (d.get("content") or "").strip()
    gold = str(item["answer"])
    sub = gold.lower() in ans.lower()
    score = f1(ans, gold)
    rows.append({"qid": qid, "type": item["question_type"], "ok": bool(sub or score >= 0.5),
                 "sub": bool(sub), "f1": round(score, 4), "answer": ans[:4000], "gold": gold,
                 "n_tokens": n, "n_evidence_sessions": len(keep_idx), "over_context": truncated})
    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=1)
    if (i + 1) % 25 == 0:
        print("[oracle] %d/%d, %.0fs, last tokens=%d evidence=%d" % (i + 1, len(targets), time.time() - t0, n, len(keep_idx)), flush=True)
over = sum(r["over_context"] for r in rows)
print("done: %d rows, %d over the context window" % (len(rows), over), flush=True)
