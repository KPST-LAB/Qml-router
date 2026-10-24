ieee-scripts

These are the scripts that produced the numbers in the first submission (v61). Each .sh at
the top level drives one reported item: it starts the engine with the flags of that run and
calls a client in bin/. Names follow the same rule as the logs in pub/test/ieee
(table, figure or section id, then the experiment, then the condition), so a log and its
driver are easy to pair; VERSIONS.txt has the full list.

There are 28 drivers, 12 clients in bin/ and two helpers at the top level (the LongMemEval
resume and recent-window variants).

If you want to re-run one, build the engine at the git tag ieee-access-submitted-v61
(third_party/llama.cpp with patches/llama.cpp-qml-session.patch, plus the vLLM and SGLang
hooks under hooks/), point the driver at your repository root, a store on the same kind of
medium as the original run, and the model file, then run it. Hosts and media for each run
are in reader.txt. Compare the new log with the frozen one line by line; the labels, order
and line counts are stable. Please keep the new log as a separate file (<name>_r1.log) and
add a TEST-LEDGER row before quoting a number.

The maintained originals live in tests/ in the repository; this folder is a frozen copy. For
the resubmission we removed the working comments and gave every file the same short header.
The logic is unchanged (we compared the files token for token, ignoring comments). The
console messages that were in Korean have been translated into English for this bundle; the
recorded logs in pub/test/ieee were already written in English, so a re-run still produces
the same log lines.
