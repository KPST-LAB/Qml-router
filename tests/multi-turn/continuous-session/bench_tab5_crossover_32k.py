#!/usr/bin/env python3
"""bench_tab5_crossover_32k.py

Purpose : Measure, per engine and configuration, the per-turn latency at each checkpoint block, the sustained throughput, and whether facts that left the window are still answerable -- the cost/benefit answer to the save-path premium
Usage   : bench_tab5_crossover_32k.py PORT ENGINE MODE TAG [SESSIONS] [TURNS] [TOKENS] [STORE_DIR]
Usage   : python3 bench_tab5_crossover_32k.py PORT ENGINE MODE TAG [SESSIONS=2]
"""
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
ENGINE = sys.argv[2]
MODE = sys.argv[3]
TAG = sys.argv[4]
SESSIONS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
TURNS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
TOKENS = int(sys.argv[7]) if len(sys.argv) > 7 else 306
STORE_DIR = sys.argv[8] if len(sys.argv) > 8 else None

WINDOW = 32768
TRIM_AT = 28000
W_TAIL = 16000
CHECKPOINTS = (100, 300, 500)
FACT_TURNS = (10, 20, 30, 40) + tuple(range(60, 461, 20))
EOT = 128009
EPID = int(os.environ.get("QML_ENGINE_PID", "0"))
SGL_EXTRA = os.environ.get("QML_SGL_EXTRA") == "1"
INJECT_BUDGET_TOK = 800

BASE = f"http://127.0.0.1:{PORT}"
TPL = os.environ.get("QML_BENCH_TEMPLATE", "llama3")
if TPL == "qwen3":
    T_SYS = "<|im_start|>system\nYou answer briefly.<|im_end|>\n"
    T_USER = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    T_EOT = "<|im_end|>"
else:
    T_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou answer briefly.<|eot_id|>"
    T_USER = "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    T_EOT = "<|eot_id|>"
_EOT_ID = None

#------------------------------------------------------------
#Func : eot_id
#End-of-turn token id of the served model (128009 for Llama 3; resolved
#------------------------------------------------------------
def eot_id():
    global _EOT_ID
    if _EOT_ID is None:
        _EOT_ID = EOT if TPL == "llama3" else post({"content": T_EOT, "add_special": False}, "/tokenize")["tokens"][-1]
    return _EOT_ID
FILLER = ("The development of transcontinental railroads in the nineteenth century "
          "reshaped commerce, migration, and the perception of distance. ")
REPS = max(1, TOKENS // 24)
SESSION_FMT = "user::x32k-%d"

#------------------------------------------------------------
#Func : typhoon
#Distinct typhoon name for session i, fact slot (0-based)
#------------------------------------------------------------
def typhoon(i, slot):
    return f"VERAX{i}{chr(65 + slot)}"

#------------------------------------------------------------
#Func : wind
#Distinct, non-guessable wind speed for session i, fact slot
#------------------------------------------------------------
def wind(i, slot):
    return 137 + 23 * i + 41 * slot

#------------------------------------------------------------
#Func : block
#One turn's new conversation text for session i at turn t (1-based)
#  in: i -- session index; t -- turn number
# out: ~TOKENS tokens; fact turns carry one weather bulletin, others filler
#------------------------------------------------------------
def block(i, t):
    if t in FACT_TURNS:
        slot = FACT_TURNS.index(t)
        fact = (f"Special weather bulletin: Typhoon {typhoon(i, slot)} made "
                f"landfall near site-{i} today with peak winds of "
                f"{wind(i, slot)} km/h. ")
        return fact + FILLER * (REPS - 1)
    return f"Continuing topic {t}. " + FILLER * REPS

#------------------------------------------------------------
#Func : question
#------------------------------------------------------------
def question(i, slot):
    return (f"According to the earlier weather bulletins, what was the exact "
            f"peak wind speed in km/h of Typhoon {typhoon(i, slot)}? "
            f"Answer with the number only.")

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(payload, path, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

_TOK = None
_NTOK_MODE = None

#------------------------------------------------------------
#Func : tok
#The HF tokenizer, needed only to detokenize archive rows (openai+qml)
#------------------------------------------------------------
def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    return _TOK

#------------------------------------------------------------
#Func : ntok
#Token count of one piece of text
#------------------------------------------------------------
def ntok(text):
    global _NTOK_MODE
    if _NTOK_MODE is None:
        try:
            tok()
            _NTOK_MODE = "hf"
        except Exception:
            _NTOK_MODE = "server"
    if _NTOK_MODE == "hf":
        return len(tok().encode(text, add_special_tokens=False))
    return len(post({"content": text, "add_special": False}, "/tokenize")["tokens"])

#------------------------------------------------------------
#Func : db
#------------------------------------------------------------
def db():
    return sqlite3.connect(f"{STORE_DIR}/qml.db", timeout=60)

#------------------------------------------------------------
#Func : log_rows
#------------------------------------------------------------
def log_rows(sid):
    con = db()
    try:
        return con.execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?",
                           (sid,)).fetchone()[0]
    finally:
        con.close()

#------------------------------------------------------------
#Func : wait_persisted
#Waits until the engine's asynchronous token-log write has landed
#------------------------------------------------------------
def wait_persisted(sid, n_before, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = log_rows(sid)
        if n != n_before and n > 0:
            return n
        time.sleep(0.02)
    print(f"[warn] {sid}: save did not arrive within {timeout}s (row count {n_before})", flush=True)
    return -1

#------------------------------------------------------------
#Func : ensure_archive_table
#------------------------------------------------------------
def ensure_archive_table(con):
    con.execute("CREATE TABLE IF NOT EXISTS qml_token_log_archive ("
                " session_id TEXT, orig_pos INTEGER, seq_id INTEGER,"
                " token_id INTEGER, archived_at INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_qml_archive_sid"
                " ON qml_token_log_archive(session_id, orig_pos)")

#------------------------------------------------------------
#Func : archive_range
#openai+qml: copies the first n_drop live-log rows into the archive
#  in: sid -- session id; n_drop -- how many leading log rows are being cut
# out: rows archived (0 when the log has fewer rows than n_drop)
#------------------------------------------------------------
def archive_range(sid, n_drop):
    con = db()
    try:
        rows = con.execute("SELECT pos, seq_id, token_id FROM qml_token_log"
                           " WHERE session_id=? ORDER BY pos LIMIT ?",
                           (sid, n_drop)).fetchall()
        if len(rows) < n_drop:
            print(f"[warn] {sid}: log rows {len(rows)} < {n_drop} to drop -- archiving less this time",
                  flush=True)
        if not rows:
            return 0
        ensure_archive_table(con)
        base = con.execute("SELECT COALESCE(MAX(orig_pos), -1) + 1"
                           " FROM qml_token_log_archive WHERE session_id=?",
                           (sid,)).fetchone()[0]
        con.executemany(
            "INSERT INTO qml_token_log_archive VALUES (?,?,?,?,strftime('%s','now'))",
            [(sid, base + k, r[1], r[2]) for k, r in enumerate(rows)])
        con.commit()
        return len(rows)
    finally:
        con.close()

#------------------------------------------------------------
#Func : recall_from_archive
#openai+qml: sentences from the DETOKENIZED archive that name the typhoon
#  in: sid -- session id; name -- typhoon name the question asks about
# out: injected text (possibly empty), built only from store content
#------------------------------------------------------------
def recall_from_archive(sid, name):
    con = db()
    try:
        try:
            ids = [r[0] for r in con.execute(
                "SELECT token_id FROM qml_token_log_archive WHERE session_id=?"
                " ORDER BY orig_pos", (sid,)).fetchall()]
        except sqlite3.OperationalError:
            return ""
    finally:
        con.close()
    if not ids:
        return ""
    text = tok().decode(ids, skip_special_tokens=True)
    hits = [s.strip() + "." for s in text.split(".") if name in s]
    out, used = [], 0
    for s in hits:
        n = ntok(s)
        if used + n > INJECT_BUDGET_TOK:
            break
        out.append(s)
        used += n
    return " ".join(out)

#------------------------------------------------------------
#Func : send_cmd
#------------------------------------------------------------
def send_cmd(cmd_type, sid, timeout=20.0):
    con = db()
    try:
        cur = con.execute(
            "INSERT INTO qml_control_cmd (target_engine, target_pid, cmd_type,"
            " session_id, created_at) VALUES ('llama.cpp', ?, ?, ?, strftime('%s','now'))",
            (EPID, cmd_type, sid))
        cid = cur.lastrowid
        con.commit()
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = con.execute("SELECT status FROM qml_control_cmd WHERE cmd_id=?",
                             (cid,)).fetchone()
            if st and st[0] in ("DONE", "FAILED"):
                return st[0]
            time.sleep(0.05)
        return "TIMEOUT"
    finally:
        con.close()

#------------------------------------------------------------
#Func : trim_llama
#Turn-aligned trim for the llama.cpp session path (gateway-equivalent)
#------------------------------------------------------------
def trim_llama(sid):
    con = db()
    try:
        rows = con.execute("SELECT pos, seq_id, token_id FROM qml_token_log"
                           " WHERE session_id=? ORDER BY pos", (sid,)).fetchall()
        n = len(rows)
        try:
            eot = eot_id()
            head_end = next(k for k, r in enumerate(rows) if r[2] == eot)
            head = rows[:head_end + 1]
            cut = next(j for j in range(max(len(head), n - W_TAIL), n)
                       if rows[j - 1][2] == eot)
        except StopIteration:
            return 0
        send_cmd("PURGE_CACHE", sid)
        send_cmd("DROP_BLOB", sid)
        for aux in ("qml_archive_fts", "qml_archive_entity", "qml_archive_cond",
                    "qml_archive_vec"):
            try:
                con.execute(f"DELETE FROM {aux} WHERE session_id=?", (sid,))
            except sqlite3.OperationalError:
                pass
        ensure_archive_table(con)
        base = con.execute("SELECT COALESCE(MAX(orig_pos), -1) + 1"
                           " FROM qml_token_log_archive WHERE session_id=?",
                           (sid,)).fetchone()[0]
        con.executemany(
            "INSERT INTO qml_token_log_archive VALUES (?,?,?,?,strftime('%s','now'))",
            [(sid, base + k, r[1], r[2]) for k, r in enumerate(rows[len(head):cut])])
        new_rows = head + rows[cut:]
        con.execute("DELETE FROM qml_token_log WHERE session_id=?", (sid,))
        con.executemany("INSERT INTO qml_token_log (session_id,pos,seq_id,token_id)"
                        " VALUES (?,?,?,?)",
                        [(sid, k, r[1], r[2]) for k, r in enumerate(new_rows)])
        con.execute("UPDATE qml_session SET n_tokens=?, last_pos=?, last_token=?,"
                    " updated_at=strftime('%s','now') WHERE session_id=?",
                    (len(new_rows), len(new_rows) - 1, new_rows[-1][2], sid))
        con.commit()
        return cut - len(head)
    finally:
        con.close()

#------------------------------------------------------------
#Func : drop_rows_after
#Removes the benchmark's own question turns from the session log
#  in: sid -- session id; keep_rows -- log length to restore
# out: None; live log truncated and the session row realigned
#------------------------------------------------------------
def drop_rows_after(sid, keep_rows):
    con = db()
    try:
        con.execute("DELETE FROM qml_token_log WHERE session_id=? AND pos>=?",
                    (sid, keep_rows))
        row = con.execute("SELECT token_id FROM qml_token_log WHERE session_id=?"
                          " ORDER BY pos DESC LIMIT 1", (sid,)).fetchone()
        if row is not None:
            con.execute("UPDATE qml_session SET n_tokens=?, last_pos=?, last_token=?,"
                        " updated_at=strftime('%s','now') WHERE session_id=?",
                        (keep_rows, keep_rows - 1, row[0], sid))
        con.commit()
    finally:
        con.close()
    send_cmd("PURGE_CACHE", sid)
    send_cmd("DROP_BLOB", sid)

MODEL = None
if ENGINE == "openai":
    MODEL = json.loads(urllib.request.urlopen(
        BASE + "/v1/models", timeout=120).read())["data"][0]["id"]

#------------------------------------------------------------
#Func : ask_llama_session
#------------------------------------------------------------
def ask_llama_session(sid, text, n_predict, first):
    body = T_USER.format(text=text)
    if first:
        body = T_SYS + body
    else:
        body = T_EOT + body
    toks = post({"content": body, "add_special": False}, "/tokenize")["tokens"]
    d = post({"prompt": toks, "n_predict": n_predict, "temperature": 0,
              "cache_prompt": True, "session_id": sid}, "/completion")
    return d.get("content", ""), int((d.get("timings") or {}).get("prompt_n") or 0)

#------------------------------------------------------------
#Func : ask_llama_plain
#------------------------------------------------------------
def ask_llama_plain(prompt, n_predict):
    d = post({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
              "cache_prompt": True}, "/completion")
    return d.get("content", ""), int((d.get("timings") or {}).get("prompt_n") or 0)

#------------------------------------------------------------
#Func : ask_openai
#------------------------------------------------------------
def ask_openai(sid, prompt, n_predict):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": n_predict,
            "temperature": 0}
    if SGL_EXTRA:
        body["extra_key"] = sid
    else:
        body["cache_salt"] = sid
    d = post(body, "/v1/completions")
    usage = d.get("usage") or {}
    cached = 0
    det = usage.get("prompt_tokens_details") or {}
    if isinstance(det, dict):
        cached = int(det.get("cached_tokens") or 0)
    return (d["choices"][0]["text"],
            max(0, int(usage.get("prompt_tokens") or 0) - cached))

QML = MODE == "qml"
if QML and STORE_DIR is None:
    sys.exit("qml mode requires the STORE_DIR argument")

live = [[] for _ in range(SESSIONS)]
lat, prefill_total = [], 0
ok_total, ask_total = 0, 0
t_wall = time.time()

for t in range(1, TURNS + 1):
    for i in range(SESSIONS):
        sid = SESSION_FMT % i
        text = block(i, t)

        if ENGINE == "llama" and QML:
            n_before = log_rows(sid)
            t0 = time.time()
            _, pf = ask_llama_session(sid, text, 8, first=(t == 1))
            lat.append((time.time() - t0) * 1000.0)
            prefill_total += pf
            n_now = wait_persisted(sid, n_before)
            if n_now > TRIM_AT:
                trim_llama(sid)
        else:
            live[i].append((text, ntok(text)))
            total = sum(c for _, c in live[i])
            if total > TRIM_AT:
                keep, acc = len(live[i]), 0
                while keep > 0 and acc + live[i][keep - 1][1] <= W_TAIL:
                    acc += live[i][keep - 1][1]
                    keep -= 1
                n_drop_tok = sum(c for _, c in live[i][:keep])
                if QML:
                    t0w = time.time()
                    while time.time() - t0w < 15.0 and log_rows(sid) < n_drop_tok:
                        time.sleep(0.02)
                    archive_range(sid, n_drop_tok)
                live[i] = live[i][keep:]
            prompt = "".join(s for s, _ in live[i])
            t0 = time.time()
            if ENGINE == "llama":
                _, pf = ask_llama_plain(prompt, 8)
            else:
                _, pf = ask_openai(sid, prompt, 8)
            lat.append((time.time() - t0) * 1000.0)
            prefill_total += pf

    if t in CHECKPOINTS:
        lo = 0 if t == CHECKPOINTS[0] else CHECKPOINTS[CHECKPOINTS.index(t) - 1]
        blk = sorted(lat[lo * SESSIONS:t * SESSIONS])
        ok, asked, detail = 0, 0, []
        for i in range(SESSIONS):
            sid = SESSION_FMT % i
            for slot, ft in enumerate(FACT_TURNS):
                if ft >= t:
                    continue
                if ENGINE == "llama" and QML:
                    in_window = False
                    tail_tok = log_rows(sid)
                    in_window = (t - ft) * TOKENS < tail_tok
                else:
                    in_window = any(typhoon(i, slot) in s for s, _ in live[i])
                if in_window:
                    continue
                asked += 1
                q = question(i, slot)
                if ENGINE == "llama" and QML:
                    keep = log_rows(sid)
                    ans, pf = ask_llama_session(sid, q, 16, first=False)
                    wait_persisted(sid, keep)
                    drop_rows_after(sid, keep)
                else:
                    inj = recall_from_archive(sid, typhoon(i, slot)) if QML else ""
                    prompt = ("Archived conversation excerpts: " + inj + "\n\n"
                              if inj else "") + "".join(s for s, _ in live[i]) + "\n" + q
                    if ENGINE == "llama":
                        ans, pf = ask_llama_plain(prompt, 16)
                    else:
                        ans, pf = ask_openai(sid, prompt, 16)
                prefill_total += pf
                good = re.search(r"\b%d\b" % wind(i, slot), ans) is not None
                ok += 1 if good else 0
                detail.append(f"s{i}{chr(65 + slot)}={'O' if good else 'X'}")
                if not good:
                    print(f"[{TAG}] wrong answer s{i}{chr(65 + slot)} "
                          f"expected={wind(i, slot)} answer={ans.strip()[:90]!r}", flush=True)
        ok_total += ok
        ask_total += asked
        print(f"[{TAG}] checkpoint turn {t}: block p50 {blk[len(blk) // 2]:,.0f} ms  "
              f"mean {statistics.mean(blk):,.0f} ms  "
              f"correct {ok}/{asked} ({' '.join(detail)})", flush=True)

wall = time.time() - t_wall
s = sorted(lat)
print(f"[{TAG}] {ENGINE}/{MODE}, sessions {SESSIONS}, turns {TURNS}, ~{TOKENS} tokens per turn, "
      f"window {WINDOW:,}: wall {wall:,.1f}s  {len(lat) / wall:,.2f} turn/s  "
      f"overall p50 {s[len(s) // 2]:,.0f} ms  GPU prefill total {prefill_total:,}  "
      f"correct total {ok_total}/{ask_total}", flush=True)
