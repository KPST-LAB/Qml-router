#!/bin/bash
#------------------------------------------------------------
#weather_per_question_copy.sh
#Purpose : Method B (per-question copy). The fed session is copied once per question and
#          each question is asked on its own copy, so no question sees another question's
#          turns. This is the procedure of the Section V-B1 runs.
#Usage   : weather_per_question_copy.sh feed PORT STORE SID N_TURNS
#          weather_per_question_copy.sh ask  PORT STORE SID COND
#          Run "feed" against a patched llama-server with archive recall off, then restart
#          the server with recall on (QML_ARCHIVE_RECALL=1, docs/API.md 5.3) and run "ask".
#          QML_ENGINE_PID must hold the server's PID (the client sends it control commands).
#------------------------------------------------------------
set -eu
B=$(cd "$(dirname "$0")/../common" && pwd)/bench_weather_longevity.py
STAGE=$1; PORT=$2; STORE=$3; SID=$4
case $STAGE in
  feed) python3 $B feed $PORT $STORE $SID $5 ;;
  ask)  python3 $B clone $PORT $STORE $SID $5
        WX_QUESTION_MODE=clone python3 $B ask $PORT $STORE $SID $5 ;;
  *)    echo "usage: $0 feed|ask PORT STORE SID N_TURNS|COND" >&2; exit 2 ;;
esac
