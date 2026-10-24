#!/usr/bin/env python3
"""lme_fullhist_resume.py

Usage   : fullhist_resume.py PORT DATASET_JSON RESULTS_JSON REF_RESULTS_JSON [N_CTX]
"""
import json, sys, time, urllib.request

PORT, DATASET, RESULTS, REF = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N_CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 131072
N_PREDICT = 256
BASE = f"http://127.0.0.1:{PORT}"
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
    lines = [f"### Session {idx} (date: {date})"]
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {t.get('content', '')}")
    return "\n".join(lines)

#------------------------------------------------------------
#Func : build_prompt
#Non-CoN reading prompt over sessions[keep_from:] (oldest dropped first)
#------------------------------------------------------------
def build_prompt(item, keep_from):
    sess = item["haystack_sessions"][keep_from:]
    dates = item["haystack_dates"][keep_from:]
    hist = "\n\n".join(session_text(i + 1, d, s) for i, (d, s) in enumerate(zip(dates, sess)))
    body = ("I will give you several history chats between you and a user. Please answer "
            "the question based on the relevant chat history.\n\nHistory Chats:\n\n"
            f"{hist}\n\nCurrent Date: {item['question_date']}\nQuestion: {item['question']}\nAnswer:")
    return (SYS_TXT + "<|start_header_id|>user<|end_header_id|>\n\n" + body +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

ds = {it["question_id"]: it for it in json.load(open(DATASET))}
rows = json.load(open(RESULTS))
done = {r["qid"] for r in rows}
targets = [r["qid"] for r in json.load(open(REF)) if r["qid"] not in done]
print(f"targets {len(targets)}, existing {len(rows)}, n_ctx {N_CTX}", flush=True)

t0 = time.time()
for i, qid in enumerate(targets):
    item = ds[qid]
    n_sess = len(item["haystack_sessions"])
    evid = set(item.get("answer_session_ids", []))
    keep_from = 0
    while True:
        prompt = build_prompt(item, keep_from)
        n = ntok(prompt)
        if n + N_PREDICT <= N_CTX or keep_from >= n_sess - 1:
            break
        keep_from += 1
    dropped_ids = item["haystack_session_ids"][:keep_from]
    dropped_evidence = sum(1 for s in dropped_ids if s in evid)
    d = post("/completion", {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                             "cache_prompt": False})
    ans = (d.get("content") or "").strip()
    gold = str(item["answer"])
    sub = gold.lower() in ans.lower()
    score = f1(ans, gold)
    rows.append({"qid": qid, "type": item["question_type"], "ok": bool(sub or score >= 0.5),
                 "sub": bool(sub), "f1": round(score, 4), "answer": ans[:4000], "gold": gold,
                 "n_tokens": n, "dropped_sessions": keep_from, "dropped_evidence": dropped_evidence})
    json.dump(rows, open(RESULTS, "w"), ensure_ascii=False, indent=1)
    print(f"[{i+1}/{len(targets)}] {qid} tokens={n} dropped={keep_from} (evidence {dropped_evidence}) "
          f"{time.time()-t0:.0f}s | {ans[:60]!r}", flush=True)
print(f"done: {len(rows)} rows in total", flush=True)
