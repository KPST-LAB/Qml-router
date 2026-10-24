# artifact

`paper/` is the supplementary material published with the article "Persistent Session-State
Management for LLM Serving: An Identity-Keyed Architecture for Durable, Engine-Agnostic
Sessions" (IEEE Access, Access-2026-49950): for every number reported in the article, the run
log, the script that produced it and the inputs that were sent to the engines.

| Path | Contents |
|---|---|
| `paper/README.txt` | layout of the bundle and how to check a number |
| `paper/VERSIONS.txt` | which log, script and data file stand behind each table and figure |
| `paper/reader.txt` | the run conditions of each log and where the reported number is in it |
| `paper/logs/` | client result logs; engine server logs, where kept, gzip-compressed |
| `paper/scripts/` | the run scripts; `scripts/bin/` holds the bench programs they call |
| `paper/data/` | inputs and result files; `data/longmemeval/` holds the LongMemEval_S answers and verdicts |
| `paper/MANIFEST.sha256` | SHA-256 of every file under `paper/` (`sha256sum -c MANIFEST.sha256` from `paper/`) |

The maintained copies of the scripts, sorted by experiment, are in `tests/paper/`. Differences
from the archive submitted with the article: `logs/reviewer1_edit_17_step1_srv_embedding.log` is
gzip-compressed here (84,339,900 bytes uncompressed) to stay below GitHub's file-size warning,
and VERSIONS.txt and reader.txt state that the article's Table 4 has no Jetson AGX Orin row and
give the input data of each Table 6 row.
