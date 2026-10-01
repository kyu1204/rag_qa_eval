#!/bin/bash
set -euo pipefail
INPUT=$(cat)

# --- oh-my-harness event logger ---
_OMH_STATE_DIR='/Users/minkyu/Documents/rag_qa_eval/.omh/state'
mkdir -p "$_OMH_STATE_DIR" 2>/dev/null || true
_OMH_HOOK_NAME="$(basename "$0")"
_OMH_EVENT="PreToolUse"
_OMH_DECISION_MODE="block"
_OMH_LOGGED=0
_log_event() {
  # omh test runs hooks as dry runs: decisions are reported, never recorded.
  if [ -n "${OMH_DRY_RUN:-}" ]; then _OMH_LOGGED=1; return 0; fi
  # Build the JSONL record entirely through jq so every string field is
  # JSON-escaped (quotes, backslashes, newlines, unicode). The previous
  # printf+%s approach corrupted the line whenever reason or any other
  # field contained these characters, and event-logger.ts silently drops
  # unparseable lines, causing event loss.
  _OMH_LOGGED=1
  local decision="${1:-allow}" reason="${2:-}" meta="${3:-}"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Meta must be a serialized JSON value (object/array/scalar); fall back
  # to no-meta when invalid so a buggy caller can't drop the event entirely.
  if [ -n "$meta" ] && ! echo "$meta" | jq -e . >/dev/null 2>&1; then
    meta=""
  fi
  if [ -n "$meta" ]; then
    jq -cn \
      --arg ts "$ts" --arg event "$_OMH_EVENT" --arg hook "$_OMH_HOOK_NAME" \
      --arg decision "$decision" --arg reason "$reason" --argjson meta "$meta" \
      '{ts:$ts,event:$event,hook:$hook,decision:$decision,reason:$reason,meta:$meta}' \
      >> "$_OMH_STATE_DIR/events.jsonl"
  else
    jq -cn \
      --arg ts "$ts" --arg event "$_OMH_EVENT" --arg hook "$_OMH_HOOK_NAME" \
      --arg decision "$decision" --arg reason "$reason" \
      '{ts:$ts,event:$event,hook:$hook,decision:$decision,reason:$reason}' \
      >> "$_OMH_STATE_DIR/events.jsonl"
  fi
}

# Emit a Claude/Codex hook decision JSON to stdout with all fields safely
# escaped. Catalog blocks should call this rather than handcrafting JSON
# via echo "{...}" — a file name or pattern containing a quote, backslash,
# or newline would otherwise produce invalid JSON that the runtime cannot
# parse as a block decision.
#
# In ask mode the same hook escalates to the user instead of hard-blocking,
# but only on runtimes that understand a permissionDecision:"ask" response.
# Claude's PreToolUse payload carries a transcript_path field; Codex's does
# not. A runtime we cannot positively identify as Claude falls through to a
# hard block, so a guardrail (e.g. TDD) is never silently downgraded to allow.
# The two requirements (Claude=ask, Codex=block) cannot coexist in one JSON —
# a legacy {decision:"block"} overrides permissionDecision:"ask" on Claude —
# so we branch on the caller instead of emitting a combined object.
_emit_decision() {
  local decision="${1:-block}" reason="${2:-}"
  if [ "${_OMH_DECISION_MODE:-block}" = "ask" ] && [ "$decision" = "block" ]; then
    if printf '%s' "${INPUT:-}" | jq -e 'has("transcript_path")' >/dev/null 2>&1; then
      jq -cn --arg reason "$reason" --arg event "$_OMH_EVENT" \
        '{hookSpecificOutput:{hookEventName:$event,permissionDecision:"ask",permissionDecisionReason:$reason}}'
      return 0
    fi
  fi
  jq -cn --arg decision "$decision" --arg reason "$reason" \
    '{decision:$decision,reason:$reason}'
}
_OMH_AWK_PATHLIB='
function omh_norm(p,    n, parts, out, i, depth, res, s) {
  n = split(p, parts, "/"); depth = 0
  for (i = 1; i <= n; i++) {
    s = parts[i]
    if (s == "" || s == ".") continue
    if (s == "..") { if (depth > 0) depth--; continue }
    out[++depth] = s
  }
  res = ""; for (i = 1; i <= depth; i++) res = res "/" out[i]
  return res == "" ? "/" : res
}
function omh_abs(cwd, t, home) {
  if (t == "" || index(t, "$") > 0 || index(t, "\140") > 0) return ""
  if (t ~ /^~(\/|$)/) { if (home == "") return ""; t = home substr(t, 2) }
  else if (t !~ /^\//) { if (cwd == "?") return ""; t = cwd "/" t }
  return omh_norm(t)
}
function omh_cd(cwd, arg, home,   a) {
  if (arg == "" || arg == "-") return "?"
  a = omh_abs(cwd, arg, home)
  return a == "" ? "?" : a
}
# cd / pushd / popd with option skipping (cd -P dir) and a directory stack.
# Returns the new cwd; the caller keeps the stack (omh_stack, omh_sp) between calls.
function omh_cd_cmd(cwd, i, home,   k, arg) {
  # pushd -n (no cd), pushd/popd +N or -N (stack rotation) and popd with options
  # change the stack in ways we do not model: unknown cwd, fail closed.
  for (k = i + 1; k <= NF; k++) {
    if ($i != "cd" && ($k == "-n" || $k ~ /^[+-][0-9]+$/)) { omh_sp = 0; return "?" }
  }
  if ($i == "popd") { return (omh_sp > 0) ? omh_stack[omh_sp--] : "?" }
  arg = ""
  for (k = i + 1; k <= NF; k++) { if ($k ~ /^-./ && $k != "-") continue; arg = $k; break }
  if ($i == "pushd") omh_stack[++omh_sp] = cwd
  return omh_cd(cwd, arg, home)
}
function omh_under(a, root) {
  return a == root || substr(a, 1, length(root) + 1) == root "/"
}'
# 1 when the command chains with ';' or newlines: a failed cd then leaves the next
# command in the OLD directory, so relative paths must also be judged from there.
_omh_seq_unsafe() { case "${1:-}" in *";"*|*$'\n'*) echo 1 ;; *) echo 0 ;; esac; }
# The project this hook belongs to: the parent of .omh/state (absolute, symlinks resolved).
_OMH_PROJECT_ROOT="$(cd "$(dirname "$(dirname "$_OMH_STATE_DIR")")" 2>/dev/null && pwd -P || pwd -P)"
_omh_tree_fingerprint() {
  local root head idx tree
  root=$(git rev-parse --show-toplevel 2>/dev/null) || { echo none; return 0; }
  head=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo empty)
  # Start from an EMPTY temporary index, never a copy of the real one: git
  # trusts cached stat data, so a file rewritten within the same second at the
  # same size would be reported unchanged (seen on Linux CI). An empty index
  # forces every file to be hashed. The real index is never touched.
  idx=$(mktemp) || { echo none; return 0; }
  rm -f "$idx"
  # Always from the repository root, whatever directory the hook runs in.
  # .omh/state is hook-owned scratch (events.jsonl grows on every hook run) and
  # must never count as a change, whether or not the user gitignored it.
  # (Naming .omh/state in an exclude pathspec makes git refuse when the path
  # is gitignored, which every omh project does; so add everything, then drop
  # it from the temporary index.)
  tree=$(GIT_INDEX_FILE="$idx" git -C "$root" add -A -- . >/dev/null 2>&1 && { GIT_INDEX_FILE="$idx" git -C "$root" rm -r -q --cached --ignore-unmatch -- .omh/state >/dev/null 2>&1 || true; } && GIT_INDEX_FILE="$idx" git -C "$root" write-tree 2>/dev/null) || tree=none
  rm -f "$idx"
  [ "$tree" = "none" ] && { echo none; return 0; }
  printf '%s:%s\n' "$head" "$tree"
}
# Gate cache. Capture the fingerprint BEFORE running the gate command and pass
# it to both helpers, so what gets recorded is the tree the command actually
# checked, not whatever it left behind (snapshot updates, formatters).
#   _omh_gate_cached <name> <ttl-seconds> <fp>  -> 0 when the last recorded pass
#                                                  for <name> is <fp> and younger than ttl.
#   _omh_gate_record <name> <fp>                -> remember <fp> as passed.
# The optional [cmd] argument becomes part of the cache key, so changing the
# command in harness.yaml never reuses a pass recorded for the old one.
_omh_gate_key() {
  local name="${1:-}" cmd="${2:-}"
  if [ -n "$cmd" ]; then printf '%s-%s' "$name" "$(printf '%s' "$cmd" | cksum | cut -d' ' -f1)"; else printf '%s' "$name"; fi
}
_omh_gate_cached() {
  local name="$1" ttl="${2:-0}" fp="${3:-none}" file now cached_fp cached_ts
  [ "$ttl" -gt 0 ] 2>/dev/null || return 1
  [ "$fp" = "none" ] && return 1
  file="${_OMH_STATE_DIR:-.omh/state}/gate-$(_omh_gate_key "$name" "${4:-}").fp"
  [ -f "$file" ] || return 1
  read -r cached_fp cached_ts < "$file" || return 1
  now=$(date +%s)
  [ "$cached_fp" = "$fp" ] && [ $((now - cached_ts)) -le "$ttl" ]
}
_omh_gate_record() {
  local name="$1" fp="${2:-none}" file
  [ "$fp" = "none" ] && return 0
  file="${_OMH_STATE_DIR:-.omh/state}/gate-$(_omh_gate_key "$name" "${3:-}").fp"
  mkdir -p "$(dirname "$file")" 2>/dev/null || true
  printf '%s %s\n' "$fp" "$(date +%s)" > "$file"
}
_omh_tokenize() {
  printf '%s\n' "${1:-}" | awk '
{ buf = buf $0 "\n" }
END {
  n = length(buf); d = 0; q[0] = 0; tok[0] = ""; cmd[0] = ""; nhd[0] = 0
  i = 1
  while (i <= n) {
    c = substr(buf, i, 1); c2 = substr(buf, i, 2)
    if (q[d] == 1) {                       # inside single quotes
      if (c == "\047") q[d] = 0; else tok[d] = tok[d] c
      i++; continue
    }
    if (q[d] == 2) {                       # inside double quotes
      if (c == "\"") { q[d] = 0; i++; continue }
      if (c == "\\" && i < n) { tok[d] = tok[d] substr(buf, i+1, 1); i += 2; continue }
      if (c2 == "$(") { d++; q[d] = 0; tok[d] = ""; cmd[d] = ""; nhd[d] = 0; i += 2; continue }
      tok[d] = tok[d] c; i++; continue
    }
    if (c == "\047") { q[d] = 1; i++; continue }
    if (c == "\"")   { q[d] = 2; i++; continue }
    if (c == "\\") { if (substr(buf, i+1, 1) != "\n") tok[d] = tok[d] substr(buf, i+1, 1); i += 2; continue }
    if (c == "#" && tok[d] == "") { while (i <= n && substr(buf, i, 1) != "\n") i++; continue }
    if (c2 == "$(") { d++; q[d] = 0; tok[d] = ""; cmd[d] = ""; nhd[d] = 0; i += 2; continue }
    if (substr(buf, i, 3) == "<<<") { i += 3; skipnext[d] = 1; continue }   # here-string: drop operator and word
    if (c2 == "<<") {                      # heredoc: remember the delimiter, drop the operator
      i += 2; strip[d, nhd[d] + 1] = 0; if (substr(buf, i, 1) == "-") { i++; strip[d, nhd[d] + 1] = 1 }
      while (substr(buf, i, 1) == " ") i++
      delim = ""
      while (i <= n) { h = substr(buf, i, 1); if (h ~ /[ \t\n;|&)]/) break; if (h != "\047" && h != "\"") delim = delim h; i++ }
      nhd[d]++; hd[d, nhd[d]] = delim; continue
    }
    if (c == ")" && d > 0) {               # end of $( ... )
      flush(d)
      if (cmd[d] != "") print cmd[d]
      d--; tok[d] = tok[d] "$(...)"; i++; continue
    }
    if (c ~ /[ \t]/) { flush(d); i++; continue }
    if (c == "&" && tok[d] ~ />$/) { tok[d] = tok[d] "&"; i++; continue }   # 2>&1, >&2
    if (c2 == "&>") { flush(d); tok[d] = "&"; i++; continue }               # &>log, &>>log
    if (c ~ /[;|&()\n]/) {                 # command separator
      flush(d)
      if (cmd[d] != "") print cmd[d]; cmd[d] = ""; skipnext[d] = 0
      i++
      if (c == "\n" && nhd[d] > 0) {       # skip heredoc bodies that start on the next line
        for (k = 1; k <= nhd[d]; k++) {
          while (i <= n) {
            j = index(substr(buf, i), "\n"); line = (j ? substr(buf, i, j-1) : substr(buf, i))
            i = (j ? i + j : n + 1); if (strip[d, k]) sub(/^\t+/, "", line)
            if (line == hd[d, k]) break
          }
        }
        nhd[d] = 0
      }
      continue
    }
    tok[d] = tok[d] c; i++
  }
  while (d >= 0) {
    flush(d)
    if (cmd[d] != "") print cmd[d]
    d--
  }
}
# Append the pending token to the current simple command, unless it is a
# redirection (2>/dev/null, >file, <in, 2>&1, &>log) or the word a bare
# redirection operator (2>, >, <, >>) applies to. Output-redirection targets
# are emitted as their own "__omh_redirect__<TAB>target" line so guards that
# care about writes (harness-guard) can see them; argv0 matchers never match
# that pseudo-command.
function flush(d,   t) {
  if (tok[d] == "") return
  if (skipnext[d]) {
    # the word an output redirection applies to: report it as a write target
    if (skipnext[d] == 2) print "__omh_redirect__\t" tok[d]
    skipnext[d] = 0; tok[d] = ""; return
  }
  if (tok[d] ~ /^[0-9]*(>>?|<|&>>?|>&)/) {
    if (tok[d] ~ /^[0-9]*(>>?|&>>?)$/) skipnext[d] = 2        # bare "> " / "2> " / "&> ": next word is written
    else if (tok[d] ~ /^<$/) skipnext[d] = 1                   # bare "< ": next word is read
    else if (tok[d] ~ /^[0-9]*(>>?|&>>?)[^&]/) {              # attached ">file" (not the ">&2" dup form)
      t = tok[d]; sub(/^[0-9]*(>>?|&>>?)/, "", t); print "__omh_redirect__\t" t
    }
    tok[d] = ""; return
  }
  cmd[d] = cmd[d] (cmd[d] == "" ? "" : "\t") tok[d]; tok[d] = ""
}'
}
_omh_simple_commands() {
  local depth="${2:-0}" kind rest
  _omh_tokenize "${1:-}" | while IFS= read -r line; do
    printf '%s\n' "$line" | awk '
BEGIN { FS = "\t" }
{
  i = 1; envp = ""
  while (i <= NF) {
    # NAME=value words are kept in the output (guards such as no-verify-guard
    # inspect GIT_CONFIG_*); every argv0 matcher skips them.
    while (i <= NF && $i ~ /^[A-Za-z_][A-Za-z0-9_]*=/) { envp = envp $i "\t"; i++ }
    if (i > NF) break
    sub(/.*\//, "", $i)          # /usr/bin/git -> git: argv0 matchers compare basenames
    f = $i
    if (f == "env") {
      i++; while (i <= NF && $i ~ /^-/) { if ($i ~ /^-(u|C|S)$/) i++; i++ }; continue
    }
    if (f == "nohup" || f == "command" || f == "exec" || f == "builtin" || f == "time") {
      i++; while (i <= NF && $i ~ /^-/) i++; continue
    }
    if (f == "nice") {
      i++; while (i <= NF && $i ~ /^-/) { if ($i == "-n") i++; i++ }; continue
    }
    if (f == "timeout") {
      i++; while (i <= NF && $i ~ /^-/) { if ($i ~ /^-(s|k)$/) i++; i++ }; i++; continue
    }
    if (f == "xargs") {
      i++; while (i <= NF && $i ~ /^-/) { if ($i ~ /^-(I|n|L|P|s|d|E|a)$/) i++; i++ }; continue
    }
    if (f == "sh" || f == "bash" || f == "zsh" || f == "dash" || f == "ksh") {
      j = i + 1
      while (j <= NF && $j ~ /^-/) {
        if ($j ~ /^-[A-Za-z]*c$/) { if (j + 1 <= NF) { gsub(/\t/, " ", envp); print "S\t" envp $(j + 1); next }; break }
        j++
      }
      break
    }
    if (f == "eval") {
      out = ""; for (k = i + 1; k <= NF; k++) out = out (k > i + 1 ? " " : "") $k
      gsub(/\t/, " ", envp); print "S\t" envp out; next
    }
    break
  }
  if (i > NF) next
  out = ""; for (k = i; k <= NF; k++) out = out (k > i ? "\t" : "") $k
  print "R\t" envp out
}' | while IFS=$'\t' read -r kind rest; do
      if [ "$kind" = "S" ] && [ "$depth" -lt 4 ]; then
        _omh_simple_commands "$rest" $((depth + 1))
      elif [ "$kind" = "S" ]; then
        printf '%s\n' "$line"
      else
        printf '%s\n' "$rest"
      fi
    done
  done
}
_omh_cmd_matches() {
  local a0="${2:-}" sc="${3:-}"
  _omh_simple_commands "${1:-}" | awk -F '	' -v a0="$a0" -v sc="$sc" '
    { i = 1
      while (i <= NF && $i ~ /^[A-Za-z_][A-Za-z0-9_]*=/) i++
      if (i <= NF && ($i == "sudo" || $i == "doas")) {
        i++; while (i <= NF && $i ~ /^-/) { if ($i ~ /^-(u|g|C|D|h|p|r|t|T|U)$/) i++; i++ }
      }
      if (i > NF || $i != a0) next
      if (sc == "") { found = 1; next }
      i++
      while (i <= NF && $i ~ /^-/) { if ($i == "-c" || $i == "-C") i++; i++ }
      if (i <= NF && $i == sc) { found = 1; next } }
    END { exit found ? 0 : 1 }'
}
_omh_cmd_has_pattern() {
  _omh_simple_commands "${1:-}" | awk -F '	' -v pat="${2:-}" '
    BEGIN { n = split(pat, p, " ") }
    { for (s = 1; s + n - 1 <= NF; s++) {
        ok = 1
        for (k = 1; k <= n; k++) if ($(s + k - 1) != p[k]) { ok = 0; break }
        if (ok) { found = 1; next } } }
    END { exit found ? 0 : 1 }'
}
trap '_OMH_EXIT_CODE=$?; if [ "$_OMH_LOGGED" -eq 0 ]; then if [ "$_OMH_EXIT_CODE" -ne 0 ]; then _log_event "error" "hook exited with code $_OMH_EXIT_CODE"; else _log_event "allow"; fi; fi' EXIT
# --- end logger ---

# Architect sessions pass through untouched; only the runner exports this.
[[ "${OMH_LOOP:-}" != "1" ]] && exit 0

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Codex apply_patch ships the whole patch in tool_input.command and names the
# files in "*** {Add|Update|Delete} File: <path>" headers. Collect those
# paths and treat them like file_path; the Bash write heuristic must NOT run
# over patch text (it would fire on words inside the diff).
PATCH_PATHS=()
if [[ "$TOOL_NAME" == "apply_patch" && -n "$COMMAND" ]]; then
  while IFS= read -r _p; do
    _p="${_p%$'\r'}"
    [[ -n "$_p" ]] && PATCH_PATHS+=("$_p")
  done < <(printf '%s\n' "$COMMAND" | sed -nE 's/^\*\*\* ((Add|Update|Delete) File|Move to): (.+)$/\3/p')
  COMMAND=""
fi
[[ -z "$FILE_PATH" && -z "$COMMAND" && ${#PATCH_PATHS[@]} -eq 0 ]] && exit 0

# Component-boundary matching: wrap both sides in slashes so 'ios' matches
# ios/App.swift and /repo/ios/... but never src/kiosk.ts.
_omh_path_under() {
  local file="/$1/" prefix="${2%/}"
  [[ "$file" == *"/$prefix/"* ]]
}

# Bash coverage: a shell command that mentions a protected path AND carries a
# write indicator is blocked. Reads (cat/grep of a work order) pass.
# ponytail: substring heuristic — a command writing elsewhere while merely
# mentioning a protected path is over-blocked; tighten to arg-level parsing if
# that ever bites. This guards a drifting loop, not a malicious one — evasion
# via split cd chains, variable indirection, pushd, or symlinks is explicitly
# out of scope: winning that arms race needs a filesystem sandbox, not a hook.
_omh_bash_writes_to() {
  local cmd="$1" target="${2%/}"
  [[ -z "$cmd" || -z "$target" ]] && return 1
  [[ "$cmd" == *"$target"* ]] || return 1
  # The path is interpolated into an ERE below: escape its metacharacters so
  # "docs/[orders" cannot break the pattern (grep error = silent bypass) and
  # "docs/o+rders" is compared literally.
  target=$(printf '%s' "$target" | sed -e 's/[][\.*^$(){}?+|]/\\&/g')
  local WRITE_OPS='(tee|mv|cp|rm|touch|truncate|sed[[:space:]]+-i[^[:space:]]*)'
  # write op targeting the protected path directly ...
  echo "$cmd" | grep -qE '(>|>>)[[:space:]]*[^|&;]*'"$target"'|(^|[^[:alnum:]_])'"$WRITE_OPS"'[[:space:]][^|&;]*'"$target" && return 0
  # ... or a cd into it followed by any write op (relative paths escape the
  # direct pattern once the cwd is inside the protected directory)
  echo "$cmd" | grep -qE '(^|[^[:alnum:]_])cd[[:space:]][^|&;]*'"$target" \
    && echo "$cmd" | grep -qE '>|(^|[^[:alnum:]_])'"$WRITE_OPS"'[[:space:]]' 
}

# Any path a tool call names: the direct file_path plus apply_patch headers.
_omh_any_path_under() {
  local prefix="$1" f
  [[ -n "$FILE_PATH" ]] && _omh_path_under "$FILE_PATH" "$prefix" && return 0
  for f in "${PATCH_PATHS[@]+"${PATCH_PATHS[@]}"}"; do
    _omh_path_under "$f" "$prefix" && return 0
  done
  return 1
}

WORK_ORDERS='docs/work-orders'
if _omh_any_path_under "$WORK_ORDERS" || _omh_bash_writes_to "$COMMAND" "$WORK_ORDERS"; then
  REASON="oh-my-harness: loop-guard — the loop must not write its own work orders. Mark the task 'BLOCKED: no work order' and move on; the architect writes work orders."
  _log_event "block" "$REASON"
  _emit_decision "block" "$REASON"
  exit 0
fi

ARCHITECT_ONLY=()
for prefix in "${ARCHITECT_ONLY[@]+"${ARCHITECT_ONLY[@]}"}"; do
  [[ -z "$prefix" ]] && continue
  if _omh_any_path_under "$prefix" || _omh_bash_writes_to "$COMMAND" "$prefix"; then
    REASON="oh-my-harness: loop-guard — $prefix is architect-only. Mark the task 'BLOCKED: architect-only path' and move on."
    _log_event "block" "$REASON"
    _emit_decision "block" "$REASON"
    exit 0
  fi
done

exit 0