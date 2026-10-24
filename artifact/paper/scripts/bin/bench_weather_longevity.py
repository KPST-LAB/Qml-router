#!/usr/bin/env python3
"""bench_weather_longevity.py

Purpose : [FTS-RECALL] single-session longevity benchmark (weather domain)
"""
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request

MODE = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
STORE = sys.argv[3] if len(sys.argv) > 3 else ""
SID = sys.argv[4] if len(sys.argv) > 4 else "weather-main"
ARG5 = sys.argv[5] if len(sys.argv) > 5 else ""
BASE = f"http://127.0.0.1:{PORT}"
EOT = 128009
EPID = int(os.environ.get("QML_ENGINE_PID", "0"))
W_LIVE = int(os.environ.get("WX_W_LIVE", "3000"))
W_TAIL = int(os.environ.get("WX_W_TAIL", "800"))

CITIES = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Suwon"]
DIRS = ["north", "south", "east", "west", "northeast", "southwest"]

FACTS = [
    (10, "Special bulletin: Typhoon Verena made landfall near Busan today with peak winds of 187 km/h.",
     [("direct",   "What was the peak wind speed when Typhoon Verena made landfall?", "187"),
      ("paraphrase", "How strong did the winds get during that typhoon that hit Busan?", "187"),
      ("indirect", "What was the name of the typhoon that made landfall near Busan?", "Verena")]),
    (50, "Special bulletin: Daegu set an all-time record high temperature of 41.3 degrees Celsius today.",
     [("direct",   "What record high temperature did Daegu set?", "41.3"),
      ("paraphrase", "How hot did it get in Daegu on its hottest day ever?", "41.3"),
      ("indirect", "Which city set an all-time record high temperature?", "Daegu")]),
    (200, "Special bulletin: a hailstorm in Gwangju dropped hailstones measuring 6.8 centimeters across.",
     [("direct",   "How large were the hailstones in the Gwangju hailstorm?", "6.8"),
      ("paraphrase", "What size did the hail reach when Gwangju was hit?", "6.8"),
      ("indirect", "Which city was hit by the hailstorm?", "Gwangju")]),
    (500, "Special bulletin: a lightning strike caused a blackout in Incheon lasting 13 hours.",
     [("direct",   "How long did the blackout in Incheon last?", "13"),
      ("paraphrase", "For how many hours was Incheon without power after the lightning strike?", "13"),
      ("indirect", "Which city had a blackout caused by a lightning strike?", "Incheon")]),
    (1000, "Special bulletin: dense fog closed Gimpo airport and 143 flights were cancelled.",
     [("direct",   "How many flights were cancelled when fog closed Gimpo airport?", "143"),
      ("paraphrase", "What was the flight cancellation count during the Gimpo fog closure?", "143"),
      ("indirect", "Which airport was closed by dense fog?", "Gimpo")]),
    (1500, "Special bulletin: PM2.5 in Seoul reached 217 micrograms per cubic meter, the worst air quality of the year.",
     [("direct",   "What PM2.5 level did Seoul reach on its worst air quality day?", "217"),
      ("paraphrase", "How high did the fine dust reading go in Seoul at its peak?", "217"),
      ("indirect", "Which pollutant reading marked the worst air quality day of the year in Seoul?", "PM2.5")]),
    (1900, "Special bulletin: flooding in Daejeon raised the river level by 3.9 meters overnight.",
     [("direct",   "By how many meters did the river level rise during the Daejeon flood?", "3.9"),
      ("paraphrase", "How much did the water rise overnight when Daejeon flooded?", "3.9"),
      ("indirect", "Which city had overnight flooding that raised its river level?", "Daejeon")]),
    (0, "Special bulletin: a snowstorm dropped 128 centimeters of snow on Suwon in a single day.",
     [("direct",   "How many centimeters of snow fell on Suwon during the snowstorm?", "128"),
      ("paraphrase", "What was the one-day snowfall total when Suwon was hit by the snowstorm?", "128"),
      ("indirect", "Which city recorded the single-day snowstorm total?", "Suwon")]),
    (0, "Special bulletin: wildfire smoke pushed the air quality index in Busan to 349.",
     [("direct",   "What air quality index did Busan reach during the wildfire smoke event?", "349"),
      ("paraphrase", "How high did Busan's air quality index climb when smoke arrived?", "349"),
      ("indirect", "What caused the air quality index spike in Busan?", "wildfire")]),
    (0, "Special bulletin: an earthquake of magnitude 5.4 damaged 862 buildings in Ulsan.",
     [("direct",   "How many buildings were damaged by the earthquake in Ulsan?", "862"),
      ("paraphrase", "What was the building damage count after the Ulsan earthquake?", "862"),
      ("indirect", "What magnitude was the earthquake that damaged buildings in Ulsan?", "5.4")]),
    (0, "Special bulletin: a landslide near Busan forced the evacuation of 214 homes.",
     [("direct",   "How many homes were evacuated because of the landslide near Busan?", "214"),
      ("paraphrase", "What was the evacuation count when the landslide struck near Busan?", "214"),
      ("indirect", "What natural disaster forced homes to be evacuated near Busan?", "landslide")]),
    (0, "Special bulletin: a drought pushed Daegu's reservoir level down to 812 million liters, a record low.",
     [("direct",   "What record-low reservoir volume did Daegu report during the drought?", "812"),
      ("paraphrase", "How many million liters remained in Daegu's reservoir at its lowest point?", "812"),
      ("indirect", "Which city reported a record-low reservoir level during a drought?", "Daegu")]),
    (0, "Special bulletin: a power plant fire left 630 households in Incheon without electricity.",
     [("direct",   "How many households lost electricity after the Incheon power plant fire?", "630"),
      ("paraphrase", "What was the household outage count from the Incheon power plant fire?", "630"),
      ("indirect", "Which city had a power plant fire that caused an outage?", "Incheon")]),
    (0, "Special bulletin: a subway signal failure in Gwangju delayed service by 96 minutes.",
     [("direct",   "How many minutes was Gwangju's subway delayed by the signal failure?", "96"),
      ("paraphrase", "What was the delay length when Gwangju's subway signal failed?", "96"),
      ("indirect", "Which city had a subway signal failure causing delays?", "Gwangju")]),
    (0, "Special bulletin: a toxic algae bloom spread 4.7 kilometers along a river near Daejeon.",
     [("direct",   "How many kilometers did the algae bloom spread near Daejeon?", "4.7"),
      ("paraphrase", "What length of river was affected by the algae bloom near Daejeon?", "4.7"),
      ("indirect", "Which city had a river affected by a toxic algae bloom?", "Daejeon")]),
    (0, "Special bulletin: a shipyard explosion in Ulsan required the evacuation of 372 workers.",
     [("direct",   "How many workers were evacuated after the Ulsan shipyard explosion?", "372"),
      ("paraphrase", "What was the worker evacuation count from the Ulsan shipyard explosion?", "372"),
      ("indirect", "Which city had a shipyard explosion requiring evacuation?", "Ulsan")]),
    (0, "Special bulletin: a hailstorm in Suwon damaged 288 vehicles overnight.",
     [("direct",   "How many vehicles were damaged by the Suwon hailstorm?", "288"),
      ("paraphrase", "What was the vehicle damage count from the overnight Suwon hailstorm?", "288"),
      ("indirect", "Which city had a hailstorm that damaged vehicles overnight?", "Suwon")]),
    (0, "Special bulletin: a subway fire in Seoul suspended service for 340 minutes.",
     [("direct",   "How many minutes was Seoul's subway service suspended after the fire?", "340"),
      ("paraphrase", "What was the suspension length when Seoul's subway caught fire?", "340"),
      ("indirect", "Which city had a subway fire that suspended service?", "Seoul")]),
    (0, "Special bulletin: a tsunami warning near Busan reported waves reaching 2.3 meters.",
     [("direct",   "How many meters did the waves reach during the Busan tsunami warning?", "2.3"),
      ("paraphrase", "What wave height was reported during the tsunami warning near Busan?", "2.3"),
      ("indirect", "Which city had a tsunami warning with reported wave heights?", "Busan")]),
    (0, "Special bulletin: a heatwave in Daegu caused 156 heat-related hospitalizations.",
     [("direct",   "How many people were hospitalized during the Daegu heatwave?", "156"),
      ("paraphrase", "What was the hospitalization count during Daegu's heatwave?", "156"),
      ("indirect", "Which city had a heatwave causing hospitalizations?", "Daegu")]),
    (0, "Special bulletin: a storm at Incheon airport delayed 421 flights.",
     [("direct",   "How many flights were delayed by the storm at Incheon airport?", "421"),
      ("paraphrase", "What was the flight delay count from the Incheon airport storm?", "421"),
      ("indirect", "Which airport had flights delayed by a storm?", "Incheon")]),
    (0, "Special bulletin: a chemical leak at a Gwangju factory forced the evacuation of 890 residents.",
     [("direct",   "How many residents were evacuated after the Gwangju chemical leak?", "890"),
      ("paraphrase", "What was the resident evacuation count from the Gwangju chemical leak?", "890"),
      ("indirect", "Which city had a factory chemical leak requiring evacuation?", "Gwangju")]),
    (0, "Special bulletin: a bridge closure near Daejeon added a 6.2 kilometer detour for commuters.",
     [("direct",   "How many kilometers was the detour caused by the Daejeon bridge closure?", "6.2"),
      ("paraphrase", "What detour distance resulted from the Daejeon bridge closure?", "6.2"),
      ("indirect", "Which city had a bridge closure causing a detour?", "Daejeon")]),
    (0, "Special bulletin: a refinery fire in Ulsan burned for 14 hours before containment.",
     [("direct",   "How many hours did the Ulsan refinery fire burn?", "14"),
      ("paraphrase", "What was the burn duration of the Ulsan refinery fire?", "14"),
      ("indirect", "Which city had a refinery fire requiring containment?", "Ulsan")]),
    (0, "Special bulletin: a flash flood in Suwon submerged 670 vehicles.",
     [("direct",   "How many vehicles were submerged by the Suwon flash flood?", "670"),
      ("paraphrase", "What was the vehicle submersion count from the Suwon flash flood?", "670"),
      ("indirect", "Which city had a flash flood that submerged vehicles?", "Suwon")]),
    (0, "Special bulletin: air traffic over Seoul was grounded for 195 minutes due to a drone incident.",
     [("direct",   "How many minutes was air traffic over Seoul grounded?", "195"),
      ("paraphrase", "What was the grounding duration during Seoul's drone incident?", "195"),
      ("indirect", "Which city had air traffic grounded due to a drone incident?", "Seoul")]),
    (0, "Special bulletin: a power outage in Busan affected 743 apartment complexes.",
     [("direct",   "How many apartment complexes were affected by the Busan power outage?", "743"),
      ("paraphrase", "What was the apartment complex count affected by the Busan power outage?", "743"),
      ("indirect", "Which city had a power outage affecting apartment complexes?", "Busan")]),
    (0, "Special bulletin: a dust storm reduced visibility in Daegu to 42 meters.",
     [("direct",   "How many meters did visibility drop to during the Daegu dust storm?", "42"),
      ("paraphrase", "What was the visibility distance during Daegu's dust storm?", "42"),
      ("indirect", "Which city had a dust storm reducing visibility?", "Daegu")]),
    (0, "Special bulletin: a gas leak near Incheon required an evacuation radius of 1.8 kilometers.",
     [("direct",   "How many kilometers was the evacuation radius for the Incheon gas leak?", "1.8"),
      ("paraphrase", "What radius was evacuated after the gas leak near Incheon?", "1.8"),
      ("indirect", "Which city had a gas leak requiring an evacuation radius?", "Incheon")]),
    (0, "Special bulletin: a partial bridge collapse in Gwangju trapped 230 vehicles.",
     [("direct",   "How many vehicles were trapped by the Gwangju bridge collapse?", "230"),
      ("paraphrase", "What was the vehicle count trapped by the Gwangju bridge collapse?", "230"),
      ("indirect", "Which city had a partial bridge collapse trapping vehicles?", "Gwangju")]),
]
_ft = os.environ.get("WX_FACT_TURNS", "")
if _ft:
    _pos = [int(x) for x in _ft.split(",") if x.strip()]
    FACTS = [(_pos[i], FACTS[i][1], FACTS[i][2]) for i in range(min(len(_pos), len(FACTS)))]
else:
    FACTS = [f for f in FACTS if f[0] > 0]
FACT_AT = {f[0]: f for f in FACTS}

#------------------------------------------------------------
#Func : report_text
#Deterministic daily report. Value ranges are constrained so no filler
#------------------------------------------------------------
def report_text(i):
    r = random.Random(1000 + i)
    city = CITIES[i % len(CITIES)]
    day = i
    hi = round(r.uniform(-8.0, 39.4), 1)
    lo = round(hi - r.uniform(3.0, 12.0), 1)
    hum = r.randint(20, 98)
    pr = round(r.uniform(0.0, 79.9), 1)
    wind = r.randint(2, 95)
    d = r.choice(DIRS)
    return (f"Daily report, day {day}, {city}: high {hi} C, low {lo} C,"
            f" humidity {hum} percent, precipitation {pr} mm,"
            f" wind {wind} km/h from the {d}.")

#------------------------------------------------------------
#Func : post
#------------------------------------------------------------
def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

#------------------------------------------------------------
#Func : db
#------------------------------------------------------------
def db():
    return sqlite3.connect(f"{STORE}/qml.db", timeout=60)

LAST_TIMINGS = {}

#------------------------------------------------------------
#Func : turn
#One live conversation turn through the engine session path
#------------------------------------------------------------
def turn(sid, text, n_predict=8):
    prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    row = db().execute("SELECT last_token FROM qml_session WHERE session_id=?",
                       (sid,)).fetchone()
    if row is None:
        prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                  "You are a weather log assistant. Acknowledge reports with one"
                  " short sentence; answer questions briefly.<|eot_id|>") + prompt
    elif row[0] != EOT:
        prompt = "<|eot_id|>" + prompt
    toks = post("/tokenize", {"content": prompt, "add_special": False})["tokens"]
    d = post("/completion", {"prompt": toks, "n_predict": n_predict,
                             "temperature": 0, "session_id": sid,
                             "cache_prompt": True})
    global LAST_TIMINGS
    LAST_TIMINGS = d.get("timings") or {}
    return d.get("content", "")

#------------------------------------------------------------
#Func : send_cmd
#------------------------------------------------------------
def send_cmd(cmd_type, sid):
    con = db()
    cur = con.execute(
        "INSERT INTO qml_control_cmd (target_engine, target_pid, cmd_type,"
        " session_id, created_at) VALUES ('llama.cpp', ?, ?, ?, strftime('%s','now'))",
        (EPID, cmd_type, sid))
    cid = cur.lastrowid
    con.commit()
    for _ in range(100):
        st = con.execute("SELECT status FROM qml_control_cmd WHERE cmd_id=?",
                         (cid,)).fetchone()
        if st and st[0] in ("DONE", "FAILED"):
            con.close()
            return st[0]
        time.sleep(0.2)
    con.close()
    return "TIMEOUT"

#------------------------------------------------------------
#Func : wait_persisted
#------------------------------------------------------------
def wait_persisted(sid, n_before, timeout=5.0):
    t0 = time.time()
    n = n_before
    while time.time() - t0 < timeout:
        n = db().execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?",
                         (sid,)).fetchone()[0]
        if n > n_before:
            return n
        time.sleep(0.01)
    print(f"[warn] {sid}: turn save did not arrive within {timeout}s"
          f" (row count {n_before} unchanged) -- skipping TAWT", flush=True)
    return -1

#------------------------------------------------------------
#Func : trim
#Gateway-equivalent TAWT trim: keep [system head + last >=W_TAIL tokens
#------------------------------------------------------------
def trim(sid):
    con = db()
    rows = con.execute("SELECT pos, seq_id, token_id FROM qml_token_log"
                       " WHERE session_id=? ORDER BY pos", (sid,)).fetchall()
    n = len(rows)
    head_end = next(i for i, r in enumerate(rows) if r[2] == EOT)
    head = rows[:head_end + 1]
    cut = next(j for j in range(max(len(head), n - W_TAIL), n) if rows[j - 1][2] == EOT)
    send_cmd("PURGE_CACHE", sid)
    send_cmd("DROP_BLOB", sid)
    con.execute("CREATE TABLE IF NOT EXISTS qml_token_log_archive ("
                " session_id TEXT, orig_pos INTEGER, seq_id INTEGER,"
                " token_id INTEGER, archived_at INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_qml_archive_sid"
                " ON qml_token_log_archive(session_id, orig_pos)")
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
                    [(sid, i, r[1], r[2]) for i, r in enumerate(new_rows)])
    con.execute("UPDATE qml_session SET n_tokens=?, last_pos=?, last_token=?,"
                " updated_at=strftime('%s','now') WHERE session_id=?",
                (len(new_rows), len(new_rows) - 1, new_rows[-1][2], sid))
    con.commit()
    con.close()
    return cut - len(head)

#------------------------------------------------------------
#Func : fnv_tag
#------------------------------------------------------------
def fnv_tag(sid_):
    h = 1469598103934665603
    for b in sid_.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")

#------------------------------------------------------------
#Func : clone_session
#Full SQL clone: session row + live log + archive + FTS (re-tagged)
#------------------------------------------------------------
def clone_session(src, dst):
    con = db()
    con.execute("DELETE FROM qml_token_log WHERE session_id=?", (dst,))
    con.execute("DELETE FROM qml_token_log_archive WHERE session_id=?", (dst,))
    con.execute("DELETE FROM qml_session WHERE session_id=?", (dst,))
    con.execute("INSERT INTO qml_token_log (session_id,pos,seq_id,token_id)"
                " SELECT ?, pos, seq_id, token_id FROM qml_token_log"
                " WHERE session_id=?", (dst, src))
    con.execute("INSERT INTO qml_token_log_archive"
                " SELECT ?, orig_pos, seq_id, token_id, archived_at"
                " FROM qml_token_log_archive WHERE session_id=?", (dst, src))
    con.execute("INSERT INTO qml_session (session_id, user_id, engine_type, seq_id,"
                " last_token, last_pos, n_tokens, owner_pid, lock_state, created_at, updated_at)"
                " SELECT ?, user_id, engine_type, seq_id, last_token, last_pos,"
                " n_tokens, 0, 'FREE', created_at, strftime('%s','now')"
                " FROM qml_session WHERE session_id=?", (dst, src))
    try:
        con.execute("INSERT INTO qml_archive_fts (content, tag, session_id, turn_pos, turn_end)"
                    " SELECT content, ?, ?, turn_pos, turn_end FROM qml_archive_fts"
                    " WHERE session_id=?", (fnv_tag(dst), dst, src))
    except sqlite3.OperationalError:
        pass
    try:
        con.execute("INSERT OR REPLACE INTO qml_archive_vec"
                    " (session_id, turn_pos, turn_end, vec)"
                    " SELECT ?, turn_pos, turn_end, vec FROM qml_archive_vec"
                    " WHERE session_id=?", (dst, src))
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()

#------------------------------------------------------------
#Func : qsid
#------------------------------------------------------------
def qsid(i, cls, condition):
    return f"{SID}-q{i}-{cls}-{condition}"

if MODE == "feed":
    n_turns = int(ARG5) if ARG5.isdigit() else 2000
    lat = []
    n_trims = 0
    t_start = time.time()
    for i in range(n_turns):
        text = FACT_AT[i][1] if i in FACT_AT else report_text(i)
        t0 = time.time()
        n_before = db().execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?",
                                (SID,)).fetchone()[0]
        turn(SID, text)
        lat.append({"turn": i, "ms": round((time.time() - t0) * 1000)})
        live = wait_persisted(SID, n_before)
        if live > W_LIVE:
            if os.environ.get("WX_CORE_TRIM", "0") == "1":
                send_cmd("PURGE_CACHE", SID)
                n_trims += 1
                lat[-1]["purge"] = 1
            else:
                archived = trim(SID)
                n_trims += 1
                lat[-1]["trim"] = archived
        if (i + 1) % 100 == 0:
            print(f"[feed] {i+1}/{n_turns} turns, {n_trims} trims,"
                  f" {time.time()-t_start:.0f}s", flush=True)
    json.dump(lat, open(f"{STORE}/latency_{SID}.json", "w"))
    arc = db().execute("SELECT COUNT(*) FROM qml_token_log_archive WHERE session_id=?",
                       (SID,)).fetchone()[0]
    print(f"[feed] done: {n_turns} turns, {n_trims} trims, archive={arc} tokens,"
          f" {time.time()-t_start:.0f}s", flush=True)

elif MODE == "clone":
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        for cls, q, gold in qs:
            clone_session(SID, qsid(fi, cls, ARG5))
    print(f"[clone] {len(FACTS)*3} question clones for condition {ARG5}", flush=True)

elif MODE == "ask":
    out = []
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        for cls, q, gold in qs:
            sid = qsid(fi, cls, ARG5)
            t0 = time.time()
            ans = turn(sid, q, n_predict=48)
            ok = gold.lower() in ans.lower()
            tg = LAST_TIMINGS
            out.append({"fact_turn": tix, "class": cls, "condition": ARG5, "gold": gold,
                        "ok": ok, "ms": round((time.time() - t0) * 1000),
                        "prompt_n": tg.get("prompt_n"), "prompt_ms": tg.get("prompt_ms"),
                        "predicted_n": tg.get("predicted_n"),
                        "predicted_ms": tg.get("predicted_ms"),
                        "answer": ans.strip()[:160]})
            print(f"[ask/{ARG5}] fact@{tix} {cls}: {'O' if ok else 'X'} ({ans.strip()[:60]!r})",
                  flush=True)
    json.dump(out, open(f"{STORE}/answers_{SID}_{ARG5}.json", "w"), indent=1)
    n_ok = sum(r["ok"] for r in out)
    print(f"[ask/{ARG5}] total {n_ok}/{len(out)}", flush=True)

elif MODE == "ask1":
    CLS_ROT = ("direct", "paraphrase", "indirect")
    n_total = int(os.environ.get("WX_N_TURNS", "2000"))
    out = []
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        cls, q, gold = [x for x in qs if x[0] == CLS_ROT[fi % 3]][0]
        t0 = time.time()
        ans = turn(SID, q, n_predict=48)
        tg = LAST_TIMINGS
        ok = gold.lower() in ans.lower()
        out.append({"fact_turn": tix, "age": n_total - tix, "class": cls,
                    "gold": gold, "ok": ok, "ms": round((time.time() - t0) * 1000),
                    "prompt_n": tg.get("prompt_n"), "prompt_ms": tg.get("prompt_ms"),
                    "predicted_n": tg.get("predicted_n"),
                    "predicted_ms": tg.get("predicted_ms"),
                    "answer": ans.strip()[:160]})
        print(f"[ask1] fact@{tix} (age {n_total-tix}) {cls}: {'O' if ok else 'X'}"
              f" {ans.strip()[:60]!r} {out[-1]['ms']}ms", flush=True)
        json.dump(out, open(f"{STORE}/answers_{SID}_ask1.json", "w"), indent=1)
    print(f"[ask1] total {sum(r['ok'] for r in out)}/{len(out)}", flush=True)

elif MODE == "single":
    n_turns = int(ARG5) if ARG5.isdigit() else 2000
    CLS_ROT = ("direct", "paraphrase", "indirect")
    ask_plan = {}
    ask_at = int(os.environ.get("WX_ASK_AT", str(n_turns)))
    for fi, (tix, stmt, qs) in enumerate(FACTS):
        cls, q, gold = [x for x in qs if x[0] == CLS_ROT[fi % 3]][0]
        ask_plan.setdefault(ask_at, []).append((fi, tix, cls, q, gold))
    lat, out, n_trims = [], [], 0
    t_start = time.time()
    for i in range(n_turns):
        text = FACT_AT[i][1] if i in FACT_AT else report_text(i)
        t0 = time.time()
        n_before = db().execute("SELECT COUNT(*) FROM qml_token_log WHERE session_id=?",
                                (SID,)).fetchone()[0]
        turn(SID, text)
        lat.append({"turn": i, "ms": round((time.time() - t0) * 1000)})
        live = wait_persisted(SID, n_before)
        if live > W_LIVE:
            lat[-1]["trim"] = trim(SID)
            n_trims += 1
        for (fi, tix, cls, q, gold) in ask_plan.get(i + 1, []):
            t1 = time.time()
            ans = turn(SID, q, n_predict=48)
            tg = LAST_TIMINGS
            ok = gold.lower() in ans.lower()
            out.append({"fact_turn": tix, "asked_at": i + 1, "age": i + 1 - tix,
                        "class": cls, "gold": gold, "ok": ok,
                        "ms": round((time.time() - t1) * 1000),
                        "prompt_n": tg.get("prompt_n"), "prompt_ms": tg.get("prompt_ms"),
                        "predicted_n": tg.get("predicted_n"),
                        "predicted_ms": tg.get("predicted_ms"),
                        "answer": ans.strip()[:160]})
            print(f"[single] turn {i+1}: fact@{tix} (age {i+1-tix}) {cls}:"
                  f" {'O' if ok else 'X'} {ans.strip()[:60]!r} {out[-1]['ms']}ms", flush=True)
            json.dump(out, open(f"{STORE}/answers_{SID}_single.json", "w"), indent=1)
        if (i + 1) % 100 == 0:
            print(f"[single] {i+1}/{n_turns} turns, {n_trims} trims,"
                  f" {time.time()-t_start:.0f}s", flush=True)
    json.dump(lat, open(f"{STORE}/latency_{SID}_single.json", "w"))
    n_ok = sum(r["ok"] for r in out)
    print(f"[single] done: {n_turns} turns, {n_trims} trims, {n_ok}/{len(out)} correct,"
          f" {time.time()-t_start:.0f}s", flush=True)

elif MODE == "report":
    rows = []
    for condition in ("on", "off"):
        try:
            rows += json.load(open(f"{STORE}/answers_{SID}_{condition}.json"))
        except FileNotFoundError:
            pass
    print("--- accuracy by fact age (turn index; higher = younger) ---")
    print(f"{'fact@turn':>10} {'age(turns)':>10} | {'ON d/p/i':>10} | {'OFF d/p/i':>10}")
    n_total = max((r["fact_turn"] for r in rows), default=0)
    for tix, stmt, qs in FACTS:
        cell = {}
        for condition in ("on", "off"):
            marks = []
            for cls in ("direct", "paraphrase", "indirect"):
                m = [r for r in rows if r["fact_turn"] == tix and r["class"] == cls
                     and r.get("condition", r.get("arm")) == condition]
                marks.append("O" if (m and m[0]["ok"]) else ("X" if m else "-"))
            cell[condition] = "/".join(marks)
        n_total_turns = int(os.environ.get("WX_N_TURNS", "2000"))
        print(f"{tix:>10} {n_total_turns-tix:>10} | {cell['on']:>10} | {cell['off']:>10}")
    for condition in ("on", "off"):
        sub = [r for r in rows if r.get("condition", r.get("arm")) == condition]
        if sub:
            print(f"condition {condition:>3}: {sum(r['ok'] for r in sub)}/{len(sub)}")
    try:
        lat = json.load(open(f"{STORE}/latency_{SID}.json"))
        q = len(lat) // 5
        print("--- per-turn latency flatness (quintiles of the feed) ---")
        for k in range(5):
            seg = [x["ms"] for x in lat[k*q:(k+1)*q]]
            seg.sort()
            print(f"turns {k*q:>5}-{(k+1)*q:>5}: p50={seg[len(seg)//2]}ms"
                  f" p95={seg[int(len(seg)*0.95)]}ms")
        trims = [x for x in lat if "trim" in x]
        print(f"trims: {len(trims)}")
    except FileNotFoundError:
        pass
else:
    sys.exit(f"unknown mode {MODE}")
