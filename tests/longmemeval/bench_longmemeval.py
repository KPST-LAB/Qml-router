"""bench_longmemeval.py

Purpose : [FTS-RECALL] LongMemEval runner for the qml archive-recall pipeline
Usage   : python3 bench_longmemeval.py MODE PORT STORE_DIR DATA_JSON COND [N]
"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

MODE, PORT, STORE, DATA, COND = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
CAP = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].isdigit() else 0
BASE = f"http://127.0.0.1:{PORT}"
EOT = 128009
SYS_TXT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           "You are a helpful assistant. Answer briefly and directly.<|eot_id|>")

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : tok
#------------------------------------------------------------
def tok(text):
    return post("/tokenize", {"content": text, "add_special": False})["tokens"]

#------------------------------------------------------------
#Func : db
#------------------------------------------------------------
def db():
    con = sqlite3.connect(f"{STORE}/qml.db", timeout=60)
    return con

#------------------------------------------------------------
#Func : sid_of
#------------------------------------------------------------
def sid_of(qid):
    return f"lme-{COND}::{qid}"

#------------------------------------------------------------
#Func : fnv_tag
#------------------------------------------------------------
def fnv_tag(sid_):
    h = 1469598103934665603
    for b in sid_.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")

#------------------------------------------------------------
#Func : f1
#------------------------------------------------------------
def f1(pred, gold):
    p = re.findall(r"[0-9a-z]+", pred.lower())
    g = re.findall(r"[0-9a-z]+", gold.lower())
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0)
    overlap = 0
    gc = {}
    for w in g:
        gc[w] = gc.get(w, 0) + 1
    pc = {}
    for w in p:
        pc[w] = pc.get(w, 0) + 1
    for w, c in pc.items():
        overlap += min(c, gc.get(w, 0))
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 0.0 if overlap == 0 else 2 * prec * rec / (prec + rec)

items = json.load(open(DATA))
_types = os.environ.get("LME_TYPES", "")
if _types:
    want = set(t.strip() for t in _types.split(",") if t.strip())
    items = [it for it in items if it["question_type"] in want]
    print(f"[filter] LME_TYPES={sorted(want)} -> {len(items)} questions", flush=True)
_qids = os.environ.get("LME_QIDS", "")
if _qids:
    want_q = set(q.strip() for q in _qids.split(",") if q.strip())
    items = [it for it in items if it["question_id"] in want_q]
    print(f"[filter] LME_QIDS -> {len(items)} questions", flush=True)
if CAP:
    items = items[:CAP]

if MODE == "ingest":
    SIDE = {}
    con = db()
    con.execute("CREATE TABLE IF NOT EXISTS qml_token_log_archive ("
                " session_id TEXT, orig_pos INTEGER, seq_id INTEGER,"
                " token_id INTEGER, archived_at INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_qml_archive_sid"
                " ON qml_token_log_archive(session_id, orig_pos)")
    sys_toks = tok(SYS_TXT)
    PREINDEX = os.environ.get("LME_PREINDEX", "1") == "1"
    DATE_INLINE = os.environ.get("LME_DATE_INLINE", "1") == "1"
    if PREINDEX:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS qml_archive_fts USING fts5("
                    " content, tag, session_id UNINDEXED, turn_pos UNINDEXED, turn_end UNINDEXED)")
    t0 = time.time()
    done = 0
    for it in items:
        sid = sid_of(it["question_id"])
        if con.execute("SELECT 1 FROM qml_session WHERE session_id=?", (sid,)).fetchone():
            continue
        pos = len(sys_toks)
        rows = []
        fts_rows = []
        ans_ranges = []
        ans_ids = set(it.get("answer_session_ids") or [])

        def _put_turn(role, body, date=None):
            global pos
            if DATE_INLINE and date:
                body = f"[{date}] {body}"
            text = (f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    f"{body}<|eot_id|>")
            toks = tok(text)
            t_start = pos
            for t in toks:
                rows.append((sid, pos, 0, t))
                pos += 1
            if PREINDEX and len(toks) >= 6:
                fts_rows.append((f"{role}\n\n{body}", fnv_tag(sid), sid, t_start, pos))

        for si, sess in enumerate(it["haystack_sessions"]):
            s_start = pos
            date = (it.get("haystack_dates") or [None] * len(it["haystack_sessions"]))[si]
            if date and not DATE_INLINE:
                _put_turn("user", f"[This conversation session takes place on {date}.]")
            for turn in sess:
                _put_turn(turn.get("role", "user"), turn.get("content", ""), date)
            sess_id_str = (it.get("haystack_session_ids") or [None] * len(it["haystack_sessions"]))[si]
            if sess_id_str in ans_ids:
                ans_ranges.append([s_start, pos])
        SIDE.setdefault(COND, {})[it["question_id"]] = ans_ranges
        con.executemany("INSERT INTO qml_token_log_archive VALUES (?,?,?,?,strftime('%s','now'))", rows)
        if PREINDEX and fts_rows:
            con.executemany("INSERT INTO qml_archive_fts"
                            " (content, tag, session_id, turn_pos, turn_end)"
                            " VALUES (?,?,?,?,?)", fts_rows)
        con.executemany("INSERT INTO qml_token_log (session_id,pos,seq_id,token_id) VALUES (?,?,?,?)",
                        [(sid, i, 0, t) for i, t in enumerate(sys_toks)])
        con.execute("INSERT INTO qml_session (session_id, user_id, engine_type, seq_id,"
                    " last_token, last_pos, n_tokens, owner_pid, lock_state, created_at, updated_at)"
                    " VALUES (?, '', 0, 0, ?, ?, ?, 0, 'FREE', strftime('%s','now'), strftime('%s','now'))",
                    (sid, sys_toks[-1], len(sys_toks) - 1, len(sys_toks)))
        con.commit()
        done += 1
        if done % 50 == 0:
            print(f"[ingest] {done} sessions, {time.time()-t0:.0f}s", flush=True)
    side_path = DATA.rsplit("/", 1)[0] + f"/answers_{COND}.json"
    if SIDE.get(COND):
        json.dump(SIDE.get(COND, {}), open(side_path, "w"))
    else:
        print(f"[ingest] 0 new sessions -- side file kept: {side_path}", flush=True)
    print(f"[ingest] done: {done} new sessions in {time.time()-t0:.0f}s; answer ranges -> {side_path}", flush=True)

elif MODE == "ask":
    import threading
    from concurrent.futures import ThreadPoolExecutor
    N_PAR = int(os.environ.get("LME_PAR", "1"))
    results = [None] * len(items)
    t0 = time.time()
    prog = {"n": 0}
    lock = threading.Lock()

    def _ask_one(i_it):
        i, it = i_it
        sid = sid_of(it["question_id"])
        q = it["question"]
        if os.environ.get("LME_QDATE"):
            q = f"Today is {it.get('question_date', '')}. {q}"
        prompt = (f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        con = db()
        lt = con.execute("SELECT last_token FROM qml_session WHERE session_id=?", (sid,)).fetchone()
        con.close()
        if lt and lt[0] == EOT:
            prompt = prompt[len("<|eot_id|>"):]
        try:
            d = post("/completion", {"prompt": tok(prompt),
                                     "n_predict": int(os.environ.get("LME_NPRED", "64")),
                                     "temperature": 0, "session_id": sid,
                                     "cache_prompt": True},
                     timeout=int(os.environ.get("LME_ASK_TIMEOUT", "900")))
            ans = d.get("content", "")
        except Exception as e:
            ans = f"[ERROR {e}]"
        gold = it["answer"] if isinstance(it["answer"], str) else str(it["answer"])
        sub = gold.lower() in ans.lower()
        score = f1(ans, gold)
        ok = sub or score >= 0.5
        results[i] = {"qid": it["question_id"], "type": it["question_type"],
                      "ok": ok, "sub": sub, "f1": round(score, 3),
                      "answer": ans.strip()[:4000], "gold": gold}
        with lock:
            prog["n"] += 1
            if prog["n"] % 25 == 0:
                done = [r for r in results if r is not None]
                acc = sum(r["ok"] for r in done) / len(done)
                print(f"[ask/{COND}] {prog['n']}/{len(items)} acc={acc:.3f}"
                      f" ({time.time()-t0:.0f}s)", flush=True)

    if N_PAR <= 1:
        for pair in enumerate(items):
            _ask_one(pair)
    else:
        with ThreadPoolExecutor(max_workers=N_PAR) as ex:
            list(ex.map(_ask_one, enumerate(items)))
    out = DATA.rsplit("/", 1)[0] + f"/results_{os.environ.get('LME_OUT_TAG', COND)}.json"
    json.dump(results, open(out, "w"), indent=1)
    by = {}
    for r in results:
        by.setdefault(r["type"], [0, 0])
        by[r["type"]][0] += 1 if r["ok"] else 0
        by[r["type"]][1] += 1
    print(f"--- summary cond={COND} (string-match metric: substring OR token-F1>=0.5) ---")
    for t, (a, n) in sorted(by.items()):
        print(f"{t:24s}: {a}/{n} = {a/n:.3f}")
    tot = sum(r["ok"] for r in results)
    print(f"{'TOTAL':24s}: {tot}/{len(results)} = {tot/len(results):.3f}")
    print(f"saved: {out}")

elif MODE == "recallk":
    side = json.load(open(DATA.rsplit("/", 1)[0] + f"/answers_{COND}.json"))
    con = db()
    hit = tot = 0
    by = {}
    for it in items:
        qid = it["question_id"]
        ranges = side.get(qid) or []
        if not ranges:
            continue
        sid = sid_of(qid)
        trow = con.execute("SELECT tag FROM qml_archive_fts WHERE session_id=? LIMIT 1",
                           (sid,)).fetchone()
        if not trow:
            continue
        words = [w for w in re.findall(r"[0-9A-Za-z-]{2,}", it["question"])][:12]
        if not words:
            continue
        q = f"tag:{trow[0]} AND (" + " OR ".join(f'"{w}"' for w in words) + ")"
        got = con.execute("SELECT turn_pos FROM qml_archive_fts WHERE qml_archive_fts"
                          " MATCH ? AND session_id=? ORDER BY rank LIMIT "
                          + os.environ.get("LME_TOPK", "3"),
                          (q, sid)).fetchall()
        ok = any(any(a <= g[0] < b for a, b in ranges) for g in got)
        hit += 1 if ok else 0
        tot += 1
        by.setdefault(it["question_type"], [0, 0])
        by[it["question_type"]][0] += 1 if ok else 0
        by[it["question_type"]][1] += 1
    print(f"--- retrieval recall@3 (reader-free) cond={COND} ---")
    for t2, (a, n) in sorted(by.items()):
        print(f"{t2:24s}: {a}/{n} = {a/n:.3f}")
    print(f"{'TOTAL':24s}: {hit}/{tot} = {hit/max(1,tot):.3f}")

elif MODE not in ("judge", "paperjudge"):
    sys.exit(f"unknown mode {MODE}")

if MODE == "judge":
    RESULTS, TAG = DATA, COND
    rows = json.load(open(RESULTS))
    qmap = {}
    if len(sys.argv) > 6 and not sys.argv[6].isdigit():
        qmap = {it["question_id"]: it["question"] for it in json.load(open(sys.argv[6]))}
    agree = {"pp": 0, "pf": 0, "fp": 0, "ff": 0}
    by = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                  "You are a strict grader. Answer with exactly one word: yes or no."
                  "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"Question: {qmap.get(r['qid'], '(unavailable)')}\n"
                  f"Gold answer: {r['gold']}\n"
                  f"Model response: {r['answer']}\n\n"
                  "Judging only whether the model response gives the same answer "
                  "to the question as the gold answer: reply yes or no.<|eot_id|>"
                  "<|start_header_id|>assistant<|end_header_id|>\n\n")
        try:
            d = post("/completion", {"prompt": tok(prompt), "n_predict": 3,
                                     "temperature": 0, "cache_prompt": False},
                     timeout=60)
            j_ok = d.get("content", "").strip().lower().startswith("yes")
        except Exception:
            j_ok = False
        r["judge_ok"] = j_ok
        key = ("p" if r["ok"] else "f") + ("p" if j_ok else "f")
        agree[key] += 1
        by.setdefault(r["type"], [0, 0])
        by[r["type"]][0] += 1 if j_ok else 0
        by[r["type"]][1] += 1
        if (i + 1) % 50 == 0:
            print(f"[judge/{TAG}] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    out = RESULTS.replace(".json", f"_judged.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"--- judge summary tag={TAG} (local 3B judge; relative use) ---")
    for t2, (a, n) in sorted(by.items()):
        print(f"{t2:24s}: {a}/{n} = {a/n:.3f}")
    tot = sum(1 for r in rows if r["judge_ok"])
    print(f"{'TOTAL':24s}: {tot}/{len(rows)} = {tot/len(rows):.3f}")
    print(f"agreement string-match vs judge: both-pass={agree['pp']} string-match-only={agree['pf']}"
          f" judge-only={agree['fp']} both-fail={agree['ff']}")
    print(f"saved: {out}")

if MODE == "paperjudge":
    RESULTS, TAG = DATA, COND
    rows = json.load(open(RESULTS))
    ds = {it["question_id"]: it for it in json.load(open(sys.argv[7]))}

    def anscheck_prompt(task, question, answer, response, abstention):
        if not abstention:
            if task in ("single-session-user", "single-session-assistant", "multi-session"):
                t = ("I will give you a question, a correct answer, and a response from a model. "
                     "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                     "If the response is equivalent to the correct answer or contains all the intermediate "
                     "steps to get the correct answer, you should also answer yes. If the response only "
                     "contains a subset of the information required by the answer, answer no. "
                     "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                     "Is the model response correct? Answer yes or no only.")
            elif task == "temporal-reasoning":
                t = ("I will give you a question, a correct answer, and a response from a model. "
                     "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                     "If the response is equivalent to the correct answer or contains all the intermediate "
                     "steps to get the correct answer, you should also answer yes. If the response only "
                     "contains a subset of the information required by the answer, answer no. In addition, "
                     "do not penalize off-by-one errors for the number of days. If the question asks for "
                     "the number of days/weeks/months, etc., and the model makes off-by-one errors "
                     "(e.g., predicting 19 days when the answer is 18), the model's response is still "
                     "correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                     "Is the model response correct? Answer yes or no only.")
            elif task == "knowledge-update":
                t = ("I will give you a question, a correct answer, and a response from a model. "
                     "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                     "If the response contains some previous information along with an updated answer, "
                     "the response should be considered as correct as long as the updated answer is the "
                     "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                     "Is the model response correct? Answer yes or no only.")
            elif task == "single-session-preference":
                t = ("I will give you a question, a rubric for desired personalized response, and a "
                     "response from a model. Please answer yes if the response satisfies the desired "
                     "response. Otherwise, answer no. The model does not need to reflect all the points "
                     "in the rubric. The response is correct as long as it recalls and utilizes the "
                     "user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
                     "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
            else:
                raise NotImplementedError(task)
        else:
            t = ("I will give you an unanswerable question, an explanation, and a response from a model. "
                 "Please answer yes if the model correctly identifies the question as unanswerable. "
                 "The model could say that the information is incomplete, or some other information is "
                 "given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
                 "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
                 "Answer yes or no only.")
        return t.format(question, answer, response)

    by, t0 = {}, time.time()
    for i, r in enumerate(rows):
        it = ds[r["qid"]]
        p = anscheck_prompt(it["question_type"], it["question"], r["gold"], r["answer"],
                            "_abs" in r["qid"])
        chat = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        try:
            d = post("/completion", {"prompt": tok(chat), "n_predict": 10,
                                     "temperature": 0, "cache_prompt": False}, timeout=120)
            j_ok = "yes" in d.get("content", "").lower()
        except Exception:
            j_ok = False
        r["paper_ok"] = j_ok
        by.setdefault(r["type"], [0, 0])
        by[r["type"]][0] += 1 if j_ok else 0
        by[r["type"]][1] += 1
        if (i + 1) % 50 == 0:
            print(f"[paperjudge/{TAG}] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    out = RESULTS.replace(".json", "_paperjudged.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"--- OFFICIAL LongMemEval protocol, tag={TAG} (judge prompts verbatim) ---")
    for t2, (a, n) in sorted(by.items()):
        print(f"{t2:24s}: {a}/{n} = {a/n:.3f}")
    tot = sum(1 for r in rows if r["paper_ok"])
    print(f"{'TOTAL':24s}: {tot}/{len(rows)} = {tot/len(rows):.3f}")
    print(f"saved: {out}")
