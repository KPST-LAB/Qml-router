#!/bin/bash
#------------------------------------------------------------
#weather_continuous_session.sh
#Purpose : Method A (conversation). The questions continue the fed conversation, without
#          per-question copies.
#          ask  = every question in order on one copy of the fed conversation
#                 (WX_QUESTION_MODE=conversation); question and answer turns stay in the
#                 history and the overflow trim runs as during the feed.
#          ask1 = one question per planted fact (the class rotates over direct,
#                 paraphrase and indirect) in the fed session itself.
#Usage   : weather_continuous_session.sh feed PORT STORE SID N_TURNS
#          weather_continuous_session.sh ask  PORT STORE SID COND
#          weather_continuous_session.sh ask1 PORT STORE SID
#          Run "feed" with archive recall off, restart the server with recall on, then
#          "ask" or "ask1". QML_ENGINE_PID must hold the server's PID.
#------------------------------------------------------------
set -eu
B=$(cd "$(dirname "$0")/../common" && pwd)/bench_weather_longevity.py
STAGE=$1; PORT=$2; STORE=$3; SID=$4
case $STAGE in
  feed) python3 $B feed $PORT $STORE $SID $5 ;;
  ask)  WX_QUESTION_MODE=conversation python3 $B ask $PORT $STORE $SID $5 ;;
  ask1) python3 $B ask1 $PORT $STORE $SID ;;
  *)    echo "usage: $0 feed|ask|ask1 PORT STORE SID [N_TURNS|COND]" >&2; exit 2 ;;
esac
