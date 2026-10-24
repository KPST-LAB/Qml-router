#!/usr/bin/env python3
"""lme_recent_window.py

Purpose : LongMemEval recent-window baseline: the same reader, the same 266 questions
Usage   : lme_recent_window.py PORT DATASET_JSON RESULTS_JSON WINDOW_TOKENS
"""
import json, sys, time, urllib.request

PORT, DATASET, RESULTS, WINDOW = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
N_PREDICT = 256
TYPES = ("multi-session", "temporal-reasoning")
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
#Func : recent_history
#Most recent turns whose Non-CoN text fits in `window` tokens, filled from the
# out: (history_text, kept_session_count, n_history_tokens)
#------------------------------------------------------------
def recent_history(item, window):
    sess, dates = item["haystack_sessions"], item["haystack_dates"]
    n = len(sess)
    flat = []
    for si in range(n - 1, -1, -1):
        for t in reversed(sess[si]):
            flat.append((si, t))

    def text_for(k, extra=None):
        """history text of the newest k turns (+ optional truncated older turn)."""
        chosen = flat[:k]
        if extra is not None:
            chosen = chosen + [extra]
        by_s = {}
        for si, t in chosen:
            by_s.setdefault(si, []).insert(0, t)
        blocks = [session_text(i + 1, dates[si], by_s[si]) for i, si in enumerate(sorted(by_s))]
        return "\n\n".join(blocks), len(by_s)

    lo, hi, best = 0, len(flat), (text_for(0)[0], 0, 0, 0)
    while lo <= hi:
        mid = (lo + hi) // 2
        h, ks = text_for(mid)
        t = ntok(h) if mid else 0
        if t <= window:
            best = (h, ks, t, mid); lo = mid + 1
        else:
            hi = mid - 1
    h, ks, t, k = best
    if k < len(flat) and window - t >= 32:
        si, nt = flat[k]
        content = str(nt.get("content", ""))
        clo, chi, cbest = 0, len(content), None
        while clo <= chi:
            mid = (clo + chi) // 2
            h2, ks2 = text_for(k, (si, dict(nt, content=content[len(content) - mid:])))
            t2 = ntok(h2)
            if t2 <= window:
                cbest = (h2, ks2, t2); clo = mid + 1
            else:
                chi = mid - 1
        if cbest is not None and cbest[2] > t:
            h, ks, t = cbest
    return h, ks, t

#------------------------------------------------------------
#Func : build_prompt
#------------------------------------------------------------
def build_prompt(item, hist):
    body = ("I will give you several history chats between you and a user. Please answer "
            "the question based on the relevant chat history.\n\nHistory Chats:\n\n"
            f"{hist}\n\nCurrent Date: {item['question_date']}\nQuestion: {item['question']}\nAnswer:")
    return (SYS_TXT + "<|start_header_id|>user<|end_header_id|>\n\n" + body +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

items = [it for it in json.load(open(DATASET)) if it["question_type"] in TYPES]
try:
    rows = json.load(open(RESULTS))
except (OSError, ValueError):
    rows = []
done = {r["qid"] for r in rows}
targets = [it for it in items if it["question_id"] not in done]
print(f"window {WINDOW}: targets {len(targets)}, existing {len(rows)}", flush=True)

t0 = time.time()
for i, item in enumerate(targets):
    qid = item["question_id"]
    evid = set(item.get("answer_session_ids", []))
    hist, kept, h_tok = recent_history(item, WINDOW)
    kept_ids = item["haystack_session_ids"][len(item["haystack_session_ids"]) - kept:]
    evid_kept = sum(1 for s in kept_ids if s in evid)
    prompt = build_prompt(item, hist)
    n = ntok(prompt)
    d = post("/completion", {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                             "cache_prompt": False})
    ans = (d.get("content") or "").strip()
    gold = str(item["answer"])
    sub = gold.lower() in ans.lower()
    score = f1(ans, gold)
    rows.append({"qid": qid, "type": item["question_type"], "ok": bool(sub or score >= 0.5),
                 "sub": bool(sub), "f1": round(score, 4), "answer": ans[:4000], "gold": gold,
                 "n_tokens": n, "history_tokens": h_tok, "window": WINDOW,
                 "kept_sessions": kept, "evidence_sessions": len(evid), "evidence_kept": evid_kept})
    json.dump(rows, open(RESULTS, "w"), ensure_ascii=False, indent=1)
    print(f"[{i+1}/{len(targets)}] {qid} hist={h_tok} kept={kept} evid={evid_kept}/{len(evid)} "
          f"{time.time()-t0:.0f}s | {ans[:60]!r}", flush=True)
print(f"done: {len(rows)} rows in total", flush=True)
