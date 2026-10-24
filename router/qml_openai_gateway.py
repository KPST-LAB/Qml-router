#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""qml_openai_gateway.py

qml-router session layer for OpenAI-compatible engines (vLLM, SGLang).

On llama.cpp the hook does everything inside the engine: the token log, the
window trim (TAWT), the archive and its retrieval (RIG). vLLM and SGLang offer
no extension point that can change a prompt, so for them this process sits in
front of the engine and does those steps over the same C store
(libqml_session_store.so, through hooks/vllm/qml_store_bindings.py):

  client --(new turn + session_id)--> gateway --(token ids + session key)--> engine
                                         |                                     |
                                   token log, TAWT,                   qml connector / backend:
                                   archive, FTS, RIG                  KV save and restore by
                                   (C core + store)                   session identity

Per turn:
  1. read the session's token log (the conversation record, owned here);
  2. RIG: search the session's archive with the new turn's words, gate the
     candidates (core qml_rig_gate) and take the chosen archived turns;
  3. send  log + recalled turns + new turn  as token ids to the engine, with the
     session id as the engine's cache key (vLLM cache_salt, SGLang extra_key);
  4. append the new turn and the answer to the token log;
  5. TAWT: when the log passes --window tokens, the core (qml_window_trim)
     moves whole turns to the archive in one transaction.

The engine module must leave the token log to this process: start the vLLM
connector or the SGLang backend with "qml_token_log": "off" in its extra config.

Retrieval settings use the names and defaults of the llama.cpp hook
(docs/API.md 5.3): QML_ARCHIVE_RECALL=1 enables RIG; QML_ARCHIVE_RECALL_TOPK
(default 3), QML_ARCHIVE_RECALL_BUDGET (1024 tokens), QML_ARCHIVE_RECALL_CONF
(0.2), QML_ARCHIVE_RECALL_NMIN (4), QML_ARCHIVE_RECALL_PAIR_TURNS (1).

API: POST /v1/chat/completions with "session_id" and the new turn as the last
user message (a "system" message is used only when the session is new). Without
"session_id" the request is passed through statelessly. The response carries a
"qml" object with what the session layer did. GET /health.

Usage:
  qml_openai_gateway.py --engine vllm --engine-url http://127.0.0.1:8000 \
      --model MODEL_NAME --tokenizer MODEL_DIR --store-db STORE/qml.db \
      --store-lmdb STORE/qml_lmdb [--window 3000] [--port 8090]
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks", "vllm"))
from qml_store_bindings import QmlStore  # noqa: E402


def _env_int(name, default):
    """Integer environment setting with a default."""
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name, default):
    """Float environment setting with a default."""
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


RECALL = os.environ.get("QML_ARCHIVE_RECALL") == "1"
TOPK = min(64, max(1, _env_int("QML_ARCHIVE_RECALL_TOPK", 3)))
BUDGET = _env_int("QML_ARCHIVE_RECALL_BUDGET", 1024)
CONF = min(1.0, max(0.0, _env_float("QML_ARCHIVE_RECALL_CONF", 0.2)))
NMIN = max(1, _env_int("QML_ARCHIVE_RECALL_NMIN", 4))
PAIR_TURNS = _env_int("QML_ARCHIVE_RECALL_PAIR_TURNS", 1) != 0
MIN_MATCH_SINGLE_HIT = 1      # same as the llama.cpp hook
INCONTEXT_TAIL = 8192         # same as the llama.cpp hook


# ---------------------------------------------------------------------------
# question-equivalence check, ported from the llama.cpp hook
# (qml_rig_idents / qml_rig_words / qml_rig_same_question): an archived row
# that is the same question asked before, without its answer, is not recalled
# ---------------------------------------------------------------------------
def _idents(text):
    """Identifiers of a text: words holding a digit and a letter ("EGR82Z", "P-107")."""
    out = []
    for w in re.split(r"[\s]+|[^\w\s-]", text or ""):
        last, digit, other = -1, False, False
        for k, c in enumerate(w):
            if c.isascii() and c.isdigit():
                digit, last = True, k
            elif c.isascii() and c.isalpha():
                other, last = True, k
            elif not c.isascii() and last < 0:
                other = True
        if digit and other and last >= 0:
            w = w[:last + 1]
            if w not in out:
                out.append(w)
    return out


def _words(text):
    """Lower-case words of two or more characters."""
    return {w.lower() for w in re.findall(r"[0-9A-Za-z\u0080-\U0010ffff]+", text or "") if len(w) >= 2}


def _same_question(row_text, question):
    """True when an archived row is only the question asked again (no answer part)."""
    if row_text is None or "\nassistant" in row_text:
        return False
    u = row_text[4:] if row_text.startswith("user") else row_text
    if "".join(u.split()) == "".join(question.split()):
        return True
    qids = _idents(question)
    if any(w not in qids for w in _idents(u)):
        return False
    rw = _words(u)
    shared = sum(1 for w in _words(question) if w in rw)
    return len(rw) <= shared + 2


def _contains(hay, needle):
    """True when the token list needle occurs in hay."""
    n = len(needle)
    if n == 0 or n > len(hay):
        return False
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return True
    return False


class SessionLayer:
    """Token log, TAWT, archive and RIG for one store, in front of one engine."""

    def __init__(self, args):
        from transformers import AutoTokenizer
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.tokenizer)
        self.eot = self._eot_id()
        self.store = QmlStore(args.lib)
        self.store.open(args.store_db, args.store_lmdb)
        self.key_field = "cache_salt" if args.engine == "vllm" else "extra_key"
        self._locks = {}
        self._locks_mu = threading.Lock()
        self._store_mu = threading.Lock()      # the C store is one process-global connection

    def _eot_id(self):
        """End-of-turn token id of the model's chat template."""
        for name in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
            tid = self.tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                return tid
        return self.tok.eos_token_id

    def _lock(self, sid):
        with self._locks_mu:
            return self._locks.setdefault(sid, threading.Lock())

    # ---- chat template in token ids ------------------------------------
    def _render(self, messages, gen):
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
        if hasattr(ids, "keys"):            # transformers >= 5 returns a BatchEncoding
            ids = ids["input_ids"]
        return list(ids)

    def head_tokens(self, system):
        """Tokens of the session's first turn (the system turn)."""
        return self._render([{"role": "system", "content": system or ""}], False)

    def turn_tokens(self, text):
        """Tokens of one user turn followed by the assistant header."""
        base = self._render([{"role": "system", "content": "x"}], False)
        full = self._render([{"role": "system", "content": "x"}, {"role": "user", "content": text}], True)
        if full[:len(base)] != base:
            raise RuntimeError("chat template: the user turn does not extend the system turn")
        return full[len(base):]

    # ---- RIG ------------------------------------------------------------
    def _index(self, sid):
        """Lazy FTS index of the archive: one document per turn, an answer merged
        into the question before it (QML_ARCHIVE_RECALL_PAIR_TURNS)."""
        pos, tok = self.store.archive_fetch(sid)
        docs, pend = [], None
        s = 0
        for i in range(len(tok)):
            if tok[i] != self.eot and i + 1 != len(tok):
                continue
            if i - s + 1 >= 6:
                text = self.tok.decode(tok[s:i + 1], skip_special_tokens=True)
                if text:
                    if PAIR_TURNS and text.startswith("assistant") and pend is not None:
                        pend[1] = pos[i] + 1
                        pend[2] += "\n" + text
                    else:
                        if pend is not None:
                            docs.append(tuple(pend))
                        pend = [pos[s], pos[i] + 1, text]
            s = i + 1
        if pend is not None:
            docs.append(tuple(pend))
        return self.store.fts_add(sid, docs)

    def recall(self, sid, question, live):
        """RIG: archived turns to put before the new turn.

        out: (token ids, picked positions, number of hits)
        """
        st = self.store.fts_status(sid)
        if st == 0:
            self._index(sid)
            st = 1
        if st != 1:
            return [], [], 0
        q_src = re.sub(r"Today is [^.]{0,60}\. ", "", question)
        query = self.store.rig_query(q_src)
        if not query:
            return [], [], 0
        hits = self.store.fts_search(sid, query, TOPK)
        if not hits:
            return [], [], 0
        best = hits[0][2] if hits[0][2] != 0.0 else -1.0
        n_eff = self.store.rig_n_words(question)
        picks, used = [], 0
        for p, e, score in hits:
            length = e - p
            if length > BUDGET or used + length > BUDGET:
                continue
            text = self.store.fts_text(sid, p)
            if _same_question(text, q_src):
                continue
            if CONF > 0.0 and not self.store.rig_gate(score, best, n_eff, NMIN, CONF):
                continue
            if len(hits) == 1 and self.store.rig_match_count(question, text) < MIN_MATCH_SINGLE_HIT:
                continue
            if any(pp == p for pp, _ in picks):
                continue
            picks.append((p, e))
            used += length
        if not picks:
            return [], [], len(hits)
        best_pick = picks[0]
        order = sorted(picks)
        order.remove(best_pick)
        order.append(best_pick)            # chronological, the best match next to the question
        tail = live[-INCONTEXT_TAIL:]
        out, picked = [], []
        for p, e in order:
            ids = list(self.store.archive_range(sid, p, e))
            if not ids or _contains(tail, ids):
                continue
            out += ids
            picked.append(p)
        return out, picked, len(hits)

    # ---- engine ---------------------------------------------------------
    def engine_complete(self, prompt_ids, max_tokens, temperature, sid):
        body = {"model": self.args.model, "prompt": prompt_ids, "max_tokens": max_tokens,
                "temperature": temperature}
        if sid:
            body[self.key_field] = sid
        req = urllib.request.Request(self.args.engine_url + "/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.args.timeout) as r:
            return json.loads(r.read())

    # ---- one turn -------------------------------------------------------
    def chat(self, body):
        msgs = body.get("messages") or []
        sid = body.get("session_id")
        max_tokens = int(body.get("max_tokens") or 256)
        temperature = float(body.get("temperature") or 0.0)
        if not sid:
            ids = self._render(msgs, True)
            d = self.engine_complete(ids, max_tokens, temperature, None)
            return d["choices"][0]["text"], d.get("usage") or {}, {"session": None}
        user = [m for m in msgs if m.get("role") == "user"]
        if not user:
            raise ValueError("no user message")
        text = user[-1].get("content") or ""
        system = next((m.get("content") for m in msgs if m.get("role") == "system"), None)
        info = {"session": sid}
        with self._lock(sid):
            t0 = time.time()
            with self._store_mu:
                live = list(self.store.tokens_get(sid))
            head = [] if live else self.head_tokens(system if system is not None
                                                    else self.args.system)
            new_turn = self.turn_tokens(text)
            inj, picked, n_hits = [], [], 0
            if RECALL and live:
                with self._store_mu:
                    inj, picked, n_hits = self.recall(sid, text, live)
            t1 = time.time()
            prompt = (live or head) + inj + new_turn
            d = self.engine_complete(prompt, max_tokens, temperature, sid)
            t2 = time.time()
            answer = d["choices"][0]["text"]
            ans_ids = self.tok.encode(answer, add_special_tokens=False) + [self.eot]
            with self._store_mu:
                if head:
                    self.store.tokens_put(sid, head, start_pos=0)
                base = len(live) if live else len(head)
                self.store.tokens_put(sid, new_turn + ans_ids, start_pos=base)
                n_log = base + len(new_turn) + len(ans_ids)
                trimmed = 0
                if self.args.window > 0 and n_log > self.args.window:
                    trimmed = self.store.window_trim(sid, self.args.window, self.eot)
            t3 = time.time()
        usage = d.get("usage") or {}
        info.update({"history_tokens": len(live or head), "injected_tokens": len(inj),
                     "recalled_pos": picked, "hits": n_hits, "prompt_tokens": len(prompt),
                     "trimmed_tokens": trimmed, "log_tokens": n_log - trimmed,
                     "cached_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                     "layer_ms": round((t1 - t0 + t3 - t2) * 1000, 1),
                     "engine_ms": round((t2 - t1) * 1000, 1)})
        return answer, usage, info


class Handler(BaseHTTPRequestHandler):
    layer = None

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "recall": RECALL, "window": self.layer.args.window})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            answer, usage, info = self.layer.chat(body)
        except Exception as e:  # report the failure to the client
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send(200, {"object": "chat.completion", "model": self.layer.args.model,
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": answer}}],
                         "usage": usage, "qml": info})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    ap.add_argument("--engine-url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", required=True, help="model directory (tokenizer and chat template)")
    ap.add_argument("--store-db", required=True)
    ap.add_argument("--store-lmdb", required=True)
    ap.add_argument("--lib", default=None, help="path to libqml_session_store.so")
    ap.add_argument("--window", type=int, default=0, help="TAWT live-window tokens (0 = no trim)")
    ap.add_argument("--system", default=None, help="system prompt of a new session without one")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    Handler.layer = SessionLayer(args)
    print(f"[qml-gateway] {args.engine} {args.engine_url} window={args.window} recall={RECALL}"
          f" topk={TOPK} budget={BUDGET} conf={CONF} pair={PAIR_TURNS} eot={Handler.layer.eot}",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
