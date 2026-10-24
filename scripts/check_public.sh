#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#------------------------------------------------------------
#check_public.sh
#Purpose : checks to run before every push to the public repository
#          1. no credential files tracked
#          2. no private IP addresses, internal paths or internal host names
#          3. no e-mail addresses other than allowed ones
#          4. SPDX license header in source files
#          5. artifact/paper/MANIFEST.sha256 verifies and lists every other file under artifact/paper
#          6. report (not fail) Hangul text outside artifact/
#Usage   : scripts/check_public.sh        (exit 1 on any failure)
#------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }

FILES=$(git ls-files)
TEXT=$(echo "$FILES" | grep -vE '\.(png|jpg|gz|pdf|zip|gguf)$')

# 1. credential files
echo "$FILES" | grep -E '(^|/)(gitinfo|\.git-credentials)$|\.info(\.key)?$' && fail "credential file tracked"

# 2. private IPs, internal paths, internal host names, GPU UUIDs
#    artifact/ holds the result records (fixed by its manifest): reported, not failed
SCAN=$(echo "$TEXT" | python3 -I -c '
import ipaddress, re, sys
ipre = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
pathre = re.compile(r"/root/jun|/home/work/|kpst4090|GPU-[0-9a-f]{8}-[0-9a-f]{4}-")
for path in sys.stdin.read().split():
    if path == "scripts/check_public.sh": continue
    try: lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError: continue
    for n, line in enumerate(lines, 1):
        hit = None
        for m in ipre.finditer(line):
            try: ip = ipaddress.ip_address(m.group(1))
            except ValueError: continue
            if ip.is_private and not (ip.is_loopback or ip.is_unspecified): hit = "ip " + m.group(1)
        if pathre.search(line): hit = "path " + pathre.search(line).group(0)
        if hit: print(("ARTIFACT " if path.startswith("artifact/") else "") + f"{path}:{n}: {hit}")
')
BAD=$(echo "$SCAN" | grep -v '^ARTIFACT ' | grep .)
[ -n "$BAD" ] && { echo "$BAD" | head -20; fail "private IP, internal path, host name or GPU UUID"; }
NART=$(echo "$SCAN" | grep -c '^ARTIFACT ')
[ "$NART" -gt 0 ] && echo "note: $NART lines in artifact/ name internal paths or addresses (result records, not changed)"

# 3. e-mail addresses (allowed: noreply, example and test domains, and the published contact in SECURITY.md, LICENSE-COMMERCIAL.md and CONTRIBUTING.md)
MAILS=$(echo "$TEXT" | xargs grep -noE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' 2>/dev/null \
        | grep -vE '@(example\.(com|org)|test\.com|users\.noreply\.github\.com)|noreply@|@llama\.cpp|^artifact/' \
        | grep -vE '^(SECURITY|LICENSE-COMMERCIAL|CONTRIBUTING)\.md:[0-9]+:jun@kpst\.co\.kr$' )   # the published contact
[ -n "$MAILS" ] && { echo "$MAILS" | head -20; fail "e-mail address"; }

# 4. SPDX headers in source files (artifact/ and tests/ copies excluded)
for f in $(echo "$FILES" | grep -E '^(core|storage|include|hooks|router|comm|scripts)/.*\.(c|h|cpp|py|sh)$'); do
  head -5 "$f" | grep -q 'SPDX-License-Identifier' || fail "missing SPDX header: $f"
done

# 5. artifact manifest
if [ -f artifact/paper/MANIFEST.sha256 ]; then
  ( cd artifact/paper && sha256sum -c --quiet MANIFEST.sha256 ) || fail "artifact manifest"
  NMAN=$(wc -l < artifact/paper/MANIFEST.sha256); NFILES=$(find artifact/paper -type f ! -name MANIFEST.sha256 | wc -l)
  [ "$NMAN" = "$NFILES" ] || fail "artifact manifest lists $NMAN files, artifact/paper has $NFILES"
fi

# 6. Hangul outside artifact/ (report only; string literals and test data may need it)
HANGUL=$(echo "$TEXT" | grep -v '^artifact/' | xargs grep -lP '[\x{AC00}-\x{D7A3}]' 2>/dev/null)
[ -n "$HANGUL" ] && { echo "note: files with Hangul text:"; echo "$HANGUL" | sed 's/^/  /'; }

[ $FAIL -eq 0 ] && echo "check_public: OK" || echo "check_public: FAILED"
exit $FAIL
