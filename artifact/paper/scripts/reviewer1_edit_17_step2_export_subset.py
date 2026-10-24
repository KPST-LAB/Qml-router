#!/usr/bin/env python3
"""reviewer1_edit_17_step2_export_subset.py

Purpose : reviewer1_edit_17 step2, part 1 - build the portable data set for the retrieval-latency comparison
Record  : TEST-LEDGER 99 (docs/3-status/TEST-LEDGER.md)
"""
import json, os, sqlite3, subprocess, time
D = "$DATA"; Q = D + "/qml-router"; OUT = Q + "/ieee-rev1/data"
DDL = ("CREATE VIRTUAL TABLE qml_archive_fts USING fts5( content, tag, session_id UNINDEXED, "
       "turn_pos UNINDEXED, turn_end UNINDEXED)")
src = sqlite3.connect("file:%s/store-lmes70/qml.db?mode=ro" % Q, uri=True, timeout=120)
dst_path = OUT + "/reviewer1_edit_17_step2_subset.db"
assert not os.path.exists(dst_path), "subset already exists: " + dst_path
dst = sqlite3.connect(dst_path)
dst.execute(DDL)
t0 = time.time()
for pfx in ("lme-msf::", "lme-trf::"):
    rows = src.execute("SELECT content, tag, session_id, turn_pos, turn_end FROM qml_archive_fts "
                       "WHERE session_id LIKE ? ORDER BY rowid", (pfx + "%",)).fetchall()
    dst.executemany("INSERT INTO qml_archive_fts (content, tag, session_id, turn_pos, turn_end) VALUES (?,?,?,?,?)", rows)
    dst.commit()
    print("%s rows %d (fact %d)" % (pfx, len(rows), sum(1 for r in rows if r[1] == "fact")), flush=True)
dst.execute("INSERT INTO qml_archive_fts(qml_archive_fts) VALUES('optimize')")
dst.commit(); dst.close()
print("subset built in %.0fs, %.1f MB" % (time.time() - t0, os.path.getsize(dst_path) / 1e6), flush=True)

data = {it["question_id"]: it for it in json.load(open(Q + "/longmemeval_s_cleaned.json"))}
qs = []
for condition, rfile in (("msf", "results_msf_final_paperjudged.json"), ("trf", "results_trf_final_paperjudged.json")):
    side = json.load(open("%s/answers_%s.json" % (Q, condition)))
    for r in json.load(open("%s/%s" % (Q, rfile))):
        qs.append({"condition": condition, "qid": r["qid"], "session_id": "lme-%s::%s" % (condition, r["qid"]),
                   "question": data[r["qid"]]["question"], "ranges": side.get(r["qid"]) or []})
p = subprocess.run([D + "/t1_query"], input="\n".join(q["question"].replace("\n", " ") for q in qs),
                   capture_output=True, text=True, timeout=180)
qmap = dict(l.split("\t", 1) for l in p.stdout.splitlines() if "\t" in l)
for q in qs:
    q["fts_query"] = qmap.get(q["question"].replace("\n", " "), "")
json.dump(qs, open(OUT + "/reviewer1_edit_17_step2_questions.json", "w"), indent=1)
print("questions %d, empty fts query %d" % (len(qs), sum(1 for q in qs if not q["fts_query"])), flush=True)
