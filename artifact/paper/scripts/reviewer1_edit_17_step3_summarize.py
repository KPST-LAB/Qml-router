#!/usr/bin/env python3
"""reviewer1_edit_17_step3_summarize.py

Purpose : Summarize step3 (lex / emb conditions, official judge) and the oracle run against v61 and step0
Usage   : reviewer1_edit_17_step3_summarize.py DATA_DIR
"""
import json, os, sys
D = sys.argv[1]
V61 = {"msf": 84, "trf": 95}

#------------------------------------------------------------
#Func : load
#------------------------------------------------------------
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

#------------------------------------------------------------
#Func : acc
#------------------------------------------------------------
def acc(rows):
    return sum(1 for r in rows if r.get("paper_ok")) if rows else None

print("| condition | multi-session (133) | temporal-reasoning (133) | note |")
print("|---|---|---|---|")
print("| v61 (lexical, engine) | %d = %.3f | %d = %.3f | submitted |" % (V61["msf"], V61["msf"] / 133, V61["trf"], V61["trf"] / 133))
s0 = {t: load("%s/reviewer1_edit_17_step0_results_%s_paperjudged.json" % (D, t)) for t in ("msf", "trf")}
print("| step0 (lexical, engine re-run) | %s | %s | same setting re-run |" % (acc(s0["msf"]), acc(s0["trf"])))
res = {}
for condition in ("lex", "emb"):
    res[condition] = {t: load("%s/reviewer1_edit_17_step3_results_%s_%s_paperjudged.json" % (D, condition, t)) for t in ("msf", "trf")}
    a, b = acc(res[condition]["msf"]), acc(res[condition]["trf"])
    label = "lexical rows, prompt rebuilt outside the engine (reproduction check)" if condition == "lex" else "embedding rows (bge-m3), same prompt construction"
    print("| step3 %s | %s | %s | %s |" % (condition, a, b, label))
orc = load("%s/reviewer1_edit_16_step1_oracle_results_paperjudged.json" % D)
if orc:
    ms = [r for r in orc if r["type"] == "multi-session"]; tr = [r for r in orc if r["type"] == "temporal-reasoning"]
    print("| oracle context (evidence sessions only) | %d | %d | full history was 33 / 43 |" % (acc(ms), acc(tr)))
    print("over-context questions:", sum(1 for r in orc if r.get("over_context")), "| evidence sessions per question: median",
          sorted(r["n_evidence_sessions"] for r in orc)[len(orc) // 2], "| prompt tokens median", sorted(r["n_tokens"] for r in orc)[len(orc) // 2])

if res["lex"]["msf"] and s0["msf"]:
    for t in ("msf", "trf"):
        a = {r["qid"]: r for r in res["lex"][t]}; b = {r["qid"]: r for r in s0[t]}
        same_ans = sum(1 for q in a if a[q]["answer"].strip() == b[q]["answer"].strip())
        same_verdict = sum(1 for q in a if bool(a[q]["paper_ok"]) == bool(b[q]["paper_ok"]))
        same_inj = sum(1 for q in a if tuple(a[q].get("engine_step0_rows_tokens") or ()) == (a[q]["n_rows"], a[q]["n_block_tokens"]))
        print("reproduction %s: injected (rows,tokens) identical %d/133, answer identical %d/133, verdict identical %d/133" % (t, same_inj, same_ans, same_verdict))
if res["emb"]["msf"] and res["lex"]["msf"]:
    for t in ("msf", "trf"):
        a = {r["qid"]: bool(r["paper_ok"]) for r in res["emb"][t]}; b = {r["qid"]: bool(r["paper_ok"]) for r in res["lex"][t]}
        print("emb vs lex %s: emb correct & lex wrong %d, lex correct & emb wrong %d" % (t, sum(1 for q in a if a[q] and not b[q]), sum(1 for q in a if b[q] and not a[q])))
