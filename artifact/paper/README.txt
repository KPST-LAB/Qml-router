Supplementary material for "Persistent Session-State Management for LLM Serving: An
Identity-Keyed Architecture for Durable, Engine-Agnostic Sessions" (IEEE Access,
Access-2026-49950).

This bundle holds, for every number reported in the article, the run log, the script that
produced it and the inputs that were sent to the engines.

  logs/          one client result log per run; engine server logs, where kept, gzip-compressed
  scripts/       the run scripts; scripts/bin/ holds the bench programs they call
  data/          the inputs (prompts, planted facts, questions, gold answers) and result files;
                 data/longmemeval/ holds the LongMemEval_S answers, verdicts and gold evidence
  VERSIONS.txt   which log, script and data file stand behind each table and figure
  reader.txt     the run conditions of each log and where the reported number is in it
  MANIFEST.sha256  SHA-256 of every file in this bundle

To check a number, find its table or figure in VERSIONS.txt, open the log, and read the
matching block in reader.txt. Paths inside scripts and logs are masked ($REPO repository
root, $DATA model and data root, $BENCH benchmark store root, $HOME, <host-ip>). Scripts
read these from the environment; the defaults are the measurement hosts' layout.

The source code (session core, llama.cpp hook, vLLM connector, SGLang backend) is released
upon publication at https://github.com/KPST-LAB/Qml-router.
