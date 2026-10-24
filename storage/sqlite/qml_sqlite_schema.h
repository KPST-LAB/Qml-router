// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.

// storage/sqlite/qml_sqlite_schema.h
//
// SQLite DDL / query strings for QML session metadata + token log.
// Heavy tensor blobs (KV state) live in LMDB — see storage/lmdb/qml_lmdb_store.h
// This header holds ONLY the schema and prepared-statement SQL text, no logic.
//
// Table layout: one common base table (qml_session, holding exactly
// qml_format.h's qml_session_meta_t fields, keyed by session_id) plus one
// extension table PER ENGINE for that engine's own extra fields (mirrors
// qml_format.h's C-inheritance idiom: base struct as first member, engine
// struct adds its own fields after it -- here, base ROW in qml_session,
// engine-specific ROW in qml_<engine>_session_ext, same session_id key).
// This keeps engines from fighting over column names/meanings in one flat
// table (e.g. llama's lmdb_key vs vLLM's lmdb_prefix) and keeps
// qml_session queryable by engine_type without any engine-specific columns.

#ifndef QML_SQLITE_SCHEMA_H
#define QML_SQLITE_SCHEMA_H

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------
//Func : QML_DDL_CREATE_SESSION
//the common base table -- one row per (user_id, session_id), mirrors
//qml_format.h's qml_session_meta_t exactly. no engine-specific column
//lives here; see the per-engine _ext tables below.
//------------------------------------------------------------
#define QML_DDL_CREATE_SESSION                                          \
    "CREATE TABLE IF NOT EXISTS qml_session ("                          \
    "  session_id  TEXT    PRIMARY KEY,"                                \
    "  user_id     TEXT    NOT NULL,"                                   \
    "  engine_type INTEGER NOT NULL DEFAULT 0,"                         \
    "  seq_id      INTEGER NOT NULL DEFAULT 0,"                         \
    "  last_token  INTEGER NOT NULL DEFAULT -1,"                        \
    "  last_pos    INTEGER NOT NULL DEFAULT -1,"                        \
    "  n_tokens    INTEGER NOT NULL DEFAULT 0,"                         \
    "  owner_pid   INTEGER NOT NULL DEFAULT 0,"                         \
    "  lock_state  TEXT    NOT NULL DEFAULT 'FREE',"                    \
    "  created_at  INTEGER NOT NULL,"                                   \
    "  updated_at  INTEGER NOT NULL"                                    \
    ");"

// owner_pid/lock_state (2026-07-29): split-brain prevention for the UDF
// control plane (see docs/udf-control-plane-proposal.md's blocker #2).
// owner_pid==0 && lock_state='FREE' means no process currently claims this
// session live. A RESTORE/PREWARM must acquire the lock (fails if another
// pid already holds it) before loading a session into live KV; MIGRATE
// releases it after evicting. NOT wired into the natural per-request
// session_id fast-path (task #16/#17/#23) -- only the explicit
// control-plane RESTORE/MIGRATE dispatch acquires/releases it, since that
// natural path predates this and reworking it is a separate, larger change
// (tracked as a known scope boundary, not silently skipped).

//------------------------------------------------------------
//Func : QML_DDL_CREATE_LLAMA_EXT
//llama.cpp-specific extension row, 1:1 with qml_session by session_id.
//mirrors storage/sqlite/qml_sqlite_store.h's qml_llama_session_meta_t
//fields beyond the base.
//------------------------------------------------------------
#define QML_DDL_CREATE_LLAMA_EXT                                        \
    "CREATE TABLE IF NOT EXISTS qml_llama_session_ext ("                \
    "  session_id  TEXT    PRIMARY KEY,"                                \
    "  model_hash  TEXT    NOT NULL DEFAULT '',"                        \
    "  lmdb_key    TEXT    NOT NULL,"                                   \
    "  state_size  INTEGER NOT NULL DEFAULT 0,"                         \
    /* slot geometry the blob was written under. n_stream is the field  \
       llama.cpp records at the head of every per-sequence state blob   \
       (= kv_unified ? 1 : n_seq_max); restoring into a context with a  \
       different value throws inside the engine. Recording it lets the  \
       restore path skip a multi-GB read it already knows will fail.    \
       0 = unknown (rows written before this column existed) -> try. */ \
    "  n_stream    INTEGER NOT NULL DEFAULT 0,"                         \
    "  n_ctx_seq   INTEGER NOT NULL DEFAULT 0"                          \
    ");"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_VLLM_EXT
//vLLM-specific extension row, 1:1 with qml_session by session_id.
//mirrors qml_format.h's qml_vllm_session_meta_t fields beyond the base.
//schema only for now -- no storage/sqlite C API reads/writes this table
//yet (that arrives with the actual vLLM adapter). Reserved so the vLLM
//adapter has a table to land on without another schema migration.
//------------------------------------------------------------
#define QML_DDL_CREATE_VLLM_EXT                                         \
    "CREATE TABLE IF NOT EXISTS qml_vllm_session_ext ("                 \
    "  session_id   TEXT    PRIMARY KEY,"                               \
    "  model_hash   TEXT    NOT NULL DEFAULT '',"                       \
    "  lmdb_prefix  TEXT    NOT NULL DEFAULT '',"                       \
    "  block_size   INTEGER NOT NULL DEFAULT 0,"                        \
    "  num_blocks   INTEGER NOT NULL DEFAULT 0"                         \
    ");"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_SGLANG_EXT
//SCHEMA ONLY, NOT WIRED YET (unlike qml_llama_session_ext/qml_vllm_session_ext,
//which both have real qml_sqlite_upsert/select_* C functions -- see
//qml_sqlite_store.h). Reason: verified by reading SGLang's actual source
//(cache_controller.py / hicache_storage.py) that HiCacheStorage's
//get/set/batch_get/batch_set/exists calls receive only opaque page-hash
//keys + prefix_keys (HiCacheStorageExtraInfo) -- NOT session_id or
//cache_salt in plaintext (those are already folded into the hash upstream,
//in RadixKey/get_hash_str, before our backend is ever called). So unlike
//vLLM's connector (which gets req.session_id directly via on_new_request),
//our SGLang KV storage backend has no session_id to write into this table
//from that call site. Populating it needs a DIFFERENT hook, most likely
//wrapping SGLang's OpenAI-compatible serving layer
//(python/sglang/srt/entrypoints/openai/serving_base.py, where
//cache_salt/session_id ARE visible pre-hash) -- not yet investigated.
//Table left reserved (like vLLM's was before task #26) so the eventual
//manifest write has somewhere to land without another migration.
//------------------------------------------------------------
#define QML_DDL_CREATE_SGLANG_EXT                                       \
    "CREATE TABLE IF NOT EXISTS qml_sglang_session_ext ("                \
    "  session_id   TEXT    PRIMARY KEY,"                               \
    "  model_hash   TEXT    NOT NULL DEFAULT '',"                       \
    "  lmdb_prefix  TEXT    NOT NULL DEFAULT '',"                       \
    "  block_size   INTEGER NOT NULL DEFAULT 0,"                        \
    "  num_blocks   INTEGER NOT NULL DEFAULT 0"                         \
    ");"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_TOKEN_LOG
//append-only per-token record — reconstructs the exact token/pos/seq
//stream a restored KV cache corresponds to (KV cache itself stores no
//token ids, only positions + tensor data). engine-agnostic as-is.
//------------------------------------------------------------
#define QML_DDL_CREATE_TOKEN_LOG                                        \
    "CREATE TABLE IF NOT EXISTS qml_token_log ("                        \
    "  session_id  TEXT    NOT NULL,"                                   \
    "  pos         INTEGER NOT NULL,"                                   \
    "  seq_id      INTEGER NOT NULL DEFAULT 0,"                         \
    "  token_id    INTEGER NOT NULL,"                                   \
    "  PRIMARY KEY (session_id, pos, seq_id)"                           \
    ");"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_TOKEN_ARCHIVE
//Append-only archive of token history that the ENGINE had to throw away.
//
//Why this exists: llama-server performs a context shift when a session reaches
//n_ctx -- it discards a middle window (common_context_seq_rm + seq_add) and
//rebuilds slot.prompt.tokens from what is left. Our snapshot save then wrote
//that shortened window over qml_token_log, so the discarded history was gone
//from OUR store as well. Measured 2026-07-30 with -c 512: a session at 318
//tokens grew, hit "context shift, n_keep = 1, n_discard = 255", and the stored
//log came back as the post-shift window with the first 255 tokens (including a
//unique marker planted at the start) permanently lost.
//
//That is backwards. The engine discards because VRAM is finite; SQLite is not.
//A session-persistence layer should be the place the full history SURVIVES.
//So before the live log is replaced by a window that is not an extension of it,
//the old rows are copied here. qml_token_log keeps meaning "the tokens the KV
//blob covers" (which is what restore and prefix matching need); this table
//keeps "everything this session ever had", for audit, cross-engine handoff and
//any future full-history rebuild.
//------------------------------------------------------------
#define QML_DDL_CREATE_TOKEN_ARCHIVE                                    \
    "CREATE TABLE IF NOT EXISTS qml_token_archive ("                    \
    "  session_id  TEXT    NOT NULL,"                                   \
    "  arch_seq    INTEGER NOT NULL,"                                   \
    "  token_id    INTEGER NOT NULL,"                                   \
    "  archived_at INTEGER NOT NULL,"                                   \
    "  PRIMARY KEY (session_id, arch_seq)"                              \
    ");"

//------------------------------------------------------------
//Func : QML_SQL_ARCHIVE_MAX_SEQ / QML_SQL_ARCHIVE_INSERT
//params: session_id | session_id, arch_seq, token_id, archived_at
//------------------------------------------------------------
#define QML_SQL_ARCHIVE_MAX_SEQ                                         \
    "SELECT COALESCE(MAX(arch_seq), -1) FROM qml_token_archive "        \
    "WHERE session_id = ?;"

#define QML_SQL_ARCHIVE_INSERT                                          \
    "INSERT OR REPLACE INTO qml_token_archive "                         \
    "  (session_id, arch_seq, token_id, archived_at) VALUES (?,?,?,?);"

//------------------------------------------------------------
//Func : QML_DDL_DROP_TOKEN_LOG_IDX
//idx_token_log_session(session_id) was REDUNDANT: qml_token_log's primary key
//is (session_id, pos, seq_id), so its implicit index already covers every
//session_id lookup -- EXPLAIN QUERY PLAN confirms both hot reads
//(SELECT ... WHERE session_id=? ORDER BY pos, and MAX(pos)) resolve through
//sqlite_autoindex_qml_token_log_1 and never touched the extra index. Keeping
//it only cost write amplification, and the token log is the write-heaviest
//table (hundreds to thousands of rows per session save). Measured on the save
//path (2000 tokens x 12 saves): 16.0ms -> 14.3ms median, i.e. ~11% faster.
//Runs as a migration so databases created before this change are fixed too.
//------------------------------------------------------------
#define QML_DDL_DROP_TOKEN_LOG_IDX                                      \
    "DROP INDEX IF EXISTS idx_token_log_session;"

//------------------------------------------------------------
//Func : QML_SQL_UPSERT_SESSION
//base table only. params (bind order): session_id, user_id, engine_type,
//                                       seq_id, last_token, last_pos,
//                                       n_tokens, created_at, updated_at
//------------------------------------------------------------
#define QML_SQL_UPSERT_SESSION                                          \
    "INSERT INTO qml_session "                                          \
    "  (session_id, user_id, engine_type, seq_id, last_token, last_pos," \
    "   n_tokens, created_at, updated_at) "                             \
    "VALUES (?,?,?,?,?,?,?,?,?) "                                       \
    "ON CONFLICT(session_id) DO UPDATE SET "                            \
    "  engine_type = excluded.engine_type,"                             \
    "  seq_id      = excluded.seq_id,"                                  \
    "  last_token  = excluded.last_token,"                              \
    "  last_pos    = excluded.last_pos,"                                \
    "  n_tokens    = excluded.n_tokens,"                                \
    "  updated_at  = excluded.updated_at;"

//------------------------------------------------------------
//Func : QML_SQL_UPSERT_LLAMA_EXT
//params (bind order): session_id, model_hash, lmdb_key, state_size, n_stream, n_ctx_seq
//------------------------------------------------------------
#define QML_SQL_UPSERT_LLAMA_EXT                                        \
    "INSERT INTO qml_llama_session_ext "                                \
    "  (session_id, model_hash, lmdb_key, state_size, n_stream, n_ctx_seq) " \
    "VALUES (?,?,?,?,?,?) "                                             \
    "ON CONFLICT(session_id) DO UPDATE SET "                            \
    "  model_hash = excluded.model_hash,"                               \
    "  lmdb_key   = excluded.lmdb_key,"                                 \
    "  state_size = excluded.state_size,"                               \
    "  n_stream   = excluded.n_stream,"                                 \
    "  n_ctx_seq  = excluded.n_ctx_seq;"

//------------------------------------------------------------
//Func : QML_SQL_UPSERT_VLLM_EXT
//params (bind order): session_id, model_hash, lmdb_prefix, block_size, num_blocks
//------------------------------------------------------------
#define QML_SQL_UPSERT_VLLM_EXT                                         \
    "INSERT INTO qml_vllm_session_ext "                                 \
    "  (session_id, model_hash, lmdb_prefix, block_size, num_blocks) "  \
    "VALUES (?,?,?,?,?) "                                               \
    "ON CONFLICT(session_id) DO UPDATE SET "                            \
    "  model_hash  = excluded.model_hash,"                              \
    "  lmdb_prefix = excluded.lmdb_prefix,"                             \
    "  block_size  = excluded.block_size,"                              \
    "  num_blocks  = excluded.num_blocks;"

//------------------------------------------------------------
//Func : QML_SQL_SELECT_VLLM_SESSION
//joins the base table with the vLLM extension table, mirrors
//QML_SQL_SELECT_LLAMA_SESSION's shape.
//params (bind order): session_id
//columns: user_id, engine_type, seq_id, last_token, last_pos, n_tokens,
//         model_hash, lmdb_prefix, block_size, num_blocks
//------------------------------------------------------------
#define QML_SQL_SELECT_VLLM_SESSION                                     \
    "SELECT s.user_id, s.engine_type, s.seq_id, s.last_token, s.last_pos," \
    "       s.n_tokens, e.model_hash, e.lmdb_prefix, e.block_size, e.num_blocks " \
    "FROM qml_session s "                                                \
    "JOIN qml_vllm_session_ext e ON e.session_id = s.session_id "       \
    "WHERE s.session_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SELECT_LLAMA_SESSION
//joins the base table with the llama extension table so callers still
//get one full qml_llama_session_meta_t out of a single query.
//params (bind order): session_id
//columns: user_id, engine_type, seq_id, last_token, last_pos, n_tokens,
//         model_hash, lmdb_key, state_size, n_stream, n_ctx_seq
//------------------------------------------------------------
#define QML_SQL_SELECT_LLAMA_SESSION                                    \
    "SELECT s.user_id, s.engine_type, s.seq_id, s.last_token, s.last_pos," \
    "       s.n_tokens, e.model_hash, e.lmdb_key, e.state_size,"         \
    "       e.n_stream, e.n_ctx_seq "                                    \
    "FROM qml_session s "                                                \
    "JOIN qml_llama_session_ext e ON e.session_id = s.session_id "      \
    "WHERE s.session_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_INSERT_TOKEN
//params (bind order): session_id, pos, seq_id, token_id
//------------------------------------------------------------
#define QML_SQL_INSERT_TOKEN                                            \
    "INSERT OR REPLACE INTO qml_token_log "                             \
    "  (session_id, pos, seq_id, token_id) VALUES (?,?,?,?);"

//------------------------------------------------------------
//Func : QML_SQL_SELECT_TOKENS_ORDERED
//params (bind order): session_id
//columns: pos, seq_id, token_id  (ORDER BY pos ASC)
//------------------------------------------------------------
#define QML_SQL_SELECT_TOKENS_ORDERED                                   \
    "SELECT pos, seq_id, token_id FROM qml_token_log "                  \
    "WHERE session_id = ? ORDER BY pos ASC;"

//------------------------------------------------------------
//Func : QML_SQL_DELETE_SESSION / QML_SQL_DELETE_LLAMA_EXT / QML_SQL_DELETE_TOKENS
//params (bind order): session_id
//used when a session is explicitly reset/evicted. all three must be run
//together (no FK cascade is set up -- kept explicit, same as before).
//------------------------------------------------------------
#define QML_SQL_DELETE_SESSION \
    "DELETE FROM qml_session WHERE session_id = ?;"

#define QML_SQL_DELETE_LLAMA_EXT \
    "DELETE FROM qml_llama_session_ext WHERE session_id = ?;"

#define QML_SQL_DELETE_VLLM_EXT \
    "DELETE FROM qml_vllm_session_ext WHERE session_id = ?;"

#define QML_SQL_DELETE_TOKENS \
    "DELETE FROM qml_token_log WHERE session_id = ?;"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_KV_BLOCK / QML_DDL_CREATE_KV_BLOCK_IDX
//LRU bookkeeping for the content-addressed KV blocks the vLLM connector
//writes into LMDB (task #56). LMDB itself has no capacity policy and the
//block keys are hashes, so without this table there is no way to know which
//bytes are cold or how much space they take -- the store grew without bound
//and qml_lmdb_delete() had no caller. last_used_us is refreshed on every
//put AND every hit, so eviction is genuine LRU rather than FIFO.
//n_bytes is stored so an eviction pass can hit a byte budget without reading LMDB.
//------------------------------------------------------------
#define QML_DDL_CREATE_KV_BLOCK                                         \
    "CREATE TABLE IF NOT EXISTS qml_kv_block ("                         \
    "  block_key    TEXT PRIMARY KEY,"                                  \
    "  n_bytes      INTEGER NOT NULL,"                                  \
    "  last_used_us INTEGER NOT NULL"                                   \
    ");"

#define QML_DDL_CREATE_KV_BLOCK_IDX                                     \
    "CREATE INDEX IF NOT EXISTS idx_qml_kv_block_lru "                  \
    "ON qml_kv_block(last_used_us);"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_KV_OWNER / QML_DDL_CREATE_KV_OWNER_IDX
//which session stored which KV block (vLLM connector blocks and markers,
//SGLang backend pages). A window trim renumbers positions and K is stored
//after RoPE, so every block a session stored before the trim can never match
//again; with this table the trim drops them at once (same rule as the
//prune of superseded segments in qml-engine). A block another owner also
//recorded (shared prefix) stays until its last owner drops it.
//------------------------------------------------------------
#define QML_DDL_CREATE_KV_OWNER                                         \
    "CREATE TABLE IF NOT EXISTS qml_kv_owner ("                         \
    "  owner      TEXT NOT NULL,"                                       \
    "  block_key  TEXT NOT NULL,"                                       \
    "  PRIMARY KEY (owner, block_key)"                                  \
    ") WITHOUT ROWID;"

#define QML_DDL_CREATE_KV_OWNER_IDX                                     \
    "CREATE INDEX IF NOT EXISTS idx_qml_kv_owner_key "                  \
    "ON qml_kv_owner(block_key);"

//------------------------------------------------------------
//Func : QML_SQL_KV_OWNER_ADD / _KEYS / _DROP / QML_SQL_KV_BLOCK_FORGET
//params: ADD (owner, block_key); KEYS and DROP (owner); FORGET (block_key)
//KEYS returns only the blocks no other owner recorded
//------------------------------------------------------------
#define QML_SQL_KV_OWNER_ADD                                            \
    "INSERT OR IGNORE INTO qml_kv_owner (owner, block_key) VALUES (?,?);"

#define QML_SQL_KV_OWNER_KEYS                                           \
    "SELECT o.block_key FROM qml_kv_owner o WHERE o.owner = ?1 "        \
    "AND NOT EXISTS (SELECT 1 FROM qml_kv_owner p "                     \
    "WHERE p.block_key = o.block_key AND p.owner <> ?1);"

#define QML_SQL_KV_OWNER_DROP                                           \
    "DELETE FROM qml_kv_owner WHERE owner = ?;"

#define QML_SQL_KV_BLOCK_FORGET                                         \
    "DELETE FROM qml_kv_block WHERE block_key = ?;"

//------------------------------------------------------------
//Func : QML_SQL_KV_BLOCK_TOUCH
//params (bind order): block_key, n_bytes, last_used_us
//------------------------------------------------------------
#define QML_SQL_KV_BLOCK_TOUCH                                          \
    "INSERT INTO qml_kv_block (block_key, n_bytes, last_used_us) "      \
    "VALUES (?,?,?) "                                                   \
    "ON CONFLICT(block_key) DO UPDATE SET last_used_us = excluded.last_used_us;"

//------------------------------------------------------------
//Func : QML_SQL_KV_BLOCK_TOTAL_BYTES / _SELECT_LRU / _DELETE
//------------------------------------------------------------
#define QML_SQL_KV_BLOCK_TOTAL_BYTES                                    \
    "SELECT COALESCE(SUM(n_bytes), 0) FROM qml_kv_block;"

// tie-breaker: rowid DESC. Batched writes/touches stamp one timestamp for a
// whole chain in insertion (= chain) order, so among equal timestamps the
// highest rowid is the TAIL of a prefix chain -- evicting it first keeps the
// front of the prefix alive longest, which is the same heuristic vLLM's own
// block pool implements by free()ing a request's blocks in reversed order
// (FreeKVCacheBlockQueue's documented eviction order).
#define QML_SQL_KV_BLOCK_SELECT_LRU                                     \
    "SELECT block_key, n_bytes FROM qml_kv_block "                      \
    "ORDER BY last_used_us ASC, rowid DESC LIMIT ?;"

#define QML_SQL_KV_BLOCK_DELETE                                         \
    "DELETE FROM qml_kv_block WHERE block_key = ?;"

//------------------------------------------------------------
//Func : QML_DDL_CREATE_CONTROL_CMD / QML_DDL_CREATE_CONTROL_CMD_BLOCKS
//UDF control-plane command queue (see docs/udf-control-plane-proposal.md).
//External management app INSERTs a PENDING row here; the target engine
//process's own background thread polls it (target_pid-scoped), claims it
//(status -> CLAIMED), executes the engine-specific action, then marks
//DONE/FAILED. payload is flattened, not a blob: session_id covers the
//scalar case common to RESTORE/MIGRATE/BACKUP_NOW; list-shaped payloads
//(PREWARM's block hash list) live in the child table instead.
//------------------------------------------------------------
#define QML_DDL_CREATE_CONTROL_CMD                                     \
    "CREATE TABLE IF NOT EXISTS qml_control_cmd ("                     \
    "  cmd_id        INTEGER PRIMARY KEY AUTOINCREMENT,"               \
    "  target_engine TEXT    NOT NULL,"                                \
    "  target_pid    INTEGER NOT NULL,"                                \
    "  cmd_type      TEXT    NOT NULL,"                                \
    "  session_id    TEXT    NOT NULL DEFAULT '',"                     \
    "  status        TEXT    NOT NULL DEFAULT 'PENDING',"              \
    "  created_at    INTEGER NOT NULL"                                 \
    ");"

#define QML_DDL_CREATE_CONTROL_CMD_IDX                                 \
    "CREATE INDEX IF NOT EXISTS idx_control_cmd_pending "               \
    "ON qml_control_cmd (target_pid, status);"

#define QML_DDL_CREATE_CONTROL_CMD_BLOCKS                               \
    "CREATE TABLE IF NOT EXISTS qml_control_cmd_blocks ("                \
    "  cmd_id      INTEGER NOT NULL,"                                   \
    "  block_hash  TEXT    NOT NULL,"                                   \
    "  FOREIGN KEY (cmd_id) REFERENCES qml_control_cmd(cmd_id)"          \
    ");"

#define QML_DDL_CREATE_CONTROL_CMD_BLOCKS_IDX                           \
    "CREATE INDEX IF NOT EXISTS idx_control_cmd_blocks_cmd "            \
    "ON qml_control_cmd_blocks (cmd_id);"

//------------------------------------------------------------
//Func : QML_SQL_INSERT_CONTROL_CMD
//params (bind order): target_engine, target_pid, cmd_type, session_id, created_at
//status defaults to 'PENDING' via DDL default, not bound here.
//caller reads sqlite3_last_insert_rowid() for the new cmd_id.
//------------------------------------------------------------
#define QML_SQL_INSERT_CONTROL_CMD                                      \
    "INSERT INTO qml_control_cmd "                                      \
    "  (target_engine, target_pid, cmd_type, session_id, created_at) "  \
    "VALUES (?,?,?,?,?);"

#define QML_SQL_INSERT_CONTROL_CMD_BLOCK                                \
    "INSERT INTO qml_control_cmd_blocks (cmd_id, block_hash) VALUES (?,?);"

//------------------------------------------------------------
//Func : QML_SQL_CLAIM_NEXT_CONTROL_CMD
//params (bind order): target_pid
//returns the oldest PENDING row (by cmd_id) for target_pid, if any.
//columns: cmd_id, target_engine, cmd_type, session_id, created_at
//caller must follow with QML_SQL_MARK_CONTROL_CMD_CLAIMED inside the same
//transaction to actually claim it (SELECT here does not itself mutate --
//see qml_sqlite_control_claim's BEGIN IMMEDIATE wrapping, which
//is what actually makes the claim atomic against concurrent claimers).
//------------------------------------------------------------
#define QML_SQL_CLAIM_NEXT_CONTROL_CMD                                  \
    "SELECT cmd_id, target_engine, cmd_type, session_id, created_at "   \
    "FROM qml_control_cmd "                                             \
    "WHERE target_pid = ? AND status = 'PENDING' "                      \
    "ORDER BY cmd_id ASC LIMIT 1;"

#define QML_SQL_MARK_CONTROL_CMD_CLAIMED                                \
    "UPDATE qml_control_cmd SET status = 'CLAIMED' WHERE cmd_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_UPDATE_CONTROL_CMD_STATUS
//params (bind order): status, cmd_id
//------------------------------------------------------------
#define QML_SQL_UPDATE_CONTROL_CMD_STATUS                                \
    "UPDATE qml_control_cmd SET status = ? WHERE cmd_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SELECT_CONTROL_CMD_BLOCKS
//params (bind order): cmd_id
//columns: block_hash
//------------------------------------------------------------
#define QML_SQL_SELECT_CONTROL_CMD_BLOCKS                                \
    "SELECT block_hash FROM qml_control_cmd_blocks WHERE cmd_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SESSION_ACQUIRE_LOCK
//params (bind order): pid, session_id, pid
//Only succeeds (changes()==1) if the session row doesn't exist as FREE-vs-
//held-by-someone-else -- i.e. WHERE lock_state='FREE' OR owner_pid=?
//(re-acquiring your own lock is a no-op success, not a conflict). If the
//session_id row doesn't exist yet at all, the caller must upsert the base
//row first (a session must be known before it can be locked).
//------------------------------------------------------------
#define QML_SQL_SESSION_ACQUIRE_LOCK                                     \
    "UPDATE qml_session SET owner_pid = ?, lock_state = 'LOCKED' "        \
    "WHERE session_id = ? AND (lock_state = 'FREE' OR owner_pid = ?);"

//------------------------------------------------------------
//Func : QML_SQL_SESSION_STEAL_LOCK
//params (bind order): new_pid, session_id, stale_pid
//Takes over a lock whose recorded owner is known to be DEAD (the caller
//verified liveness with kill(pid, 0) first). The WHERE clause re-checks the
//stale owner so a concurrent take-over by a THIRD process between the
//liveness check and this UPDATE loses cleanly (changes()==0) instead of
//silently stealing a lock a live process just acquired.
//------------------------------------------------------------
#define QML_SQL_SESSION_STEAL_LOCK                                       \
    "UPDATE qml_session SET owner_pid = ?, lock_state = 'LOCKED' "        \
    "WHERE session_id = ? AND owner_pid = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SESSION_RELEASE_LOCK
//params (bind order): session_id, pid
//Only releases if currently owned by this exact pid -- prevents one
//process accidentally releasing a lock it never held.
//------------------------------------------------------------
#define QML_SQL_SESSION_RELEASE_LOCK                                     \
    "UPDATE qml_session SET owner_pid = 0, lock_state = 'FREE' "         \
    "WHERE session_id = ? AND owner_pid = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SESSION_GET_LOCK
//params (bind order): session_id
//columns: owner_pid, lock_state
//------------------------------------------------------------
#define QML_SQL_SESSION_GET_LOCK                                         \
    "SELECT owner_pid, lock_state FROM qml_session WHERE session_id = ?;"

//------------------------------------------------------------
//Func : QML_SQL_SELECT_LAST_POS
//params (bind order): session_id
//columns: last_pos
//backs qml_sqlite_last_pos -- lets a cross-engine continuation
//append qml_token_log rows starting at last_pos+1 instead of always
//resetting to pos 0 (see docs/udf-control-plane-proposal.md's position-
//offset-precision note). qml_session.last_pos already exists and is
//already maintained by every engine's own upsert path -- this just reads
//it back through a dedicated, engine-agnostic query rather than the
//full engine-specific SELECT_*_SESSION joins.
//------------------------------------------------------------
#define QML_SQL_SELECT_LAST_POS                                          \
    "SELECT last_pos FROM qml_session WHERE session_id = ?;"

#ifdef __cplusplus
}
#endif

#endif // QML_SQLITE_SCHEMA_H
