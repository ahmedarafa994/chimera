#!/bin/bash
VENDOR="$1"
AGENT="$2"
SESSION_ID="$3"
ANSWER="$4"

SUBAGENTS_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$SUBAGENTS_DIR/$AGENT"
LOG_FILE="$SUBAGENTS_DIR/subagent.log"
TEMP_OUTPUT=$(mktemp)

cd "$AGENT_DIR"

if [ "$VENDOR" = "codex" ]; then
  # CODEX: use resume command with session_id
  # ISOLATION: CODEX_HOME points to .codex subdir with symlinked auth.json
  
  # Append to log (same session)
  echo "=== [$AGENT] RESUME $(date +%H:%M:%S) ===" >> "$LOG_FILE"
  
  CODEX_HOME="$AGENT_DIR/.codex" codex exec --dangerously-bypass-approvals-and-sandbox \
    resume "$SESSION_ID" "$ANSWER" 2> >(tee "$TEMP_OUTPUT" >> "$LOG_FILE")
  
  # Extract new session_id if provided
  NEW_SESSION_ID=$(sed 's/\x1b\[[0-9;]*m//g' "$TEMP_OUTPUT" | grep -oE "session id: [0-9a-f-]+" | head -1 | cut -d' ' -f3)
  
else
  # CLAUDE: Resume session - append to existing log file
  # ISOLATION: --setting-sources "" blocks all CLAUDE.md files
  # NOTE: Terminal is already open from start.sh, watching the log file
  
  # Find the latest session log file (same one that Terminal is watching)
  SESSIONS_DIR="$AGENT_DIR/sessions"
  LOG_FILE=$(ls -t "$SESSIONS_DIR"/*.jsonl 2>/dev/null | head -1)
  
  # If no log file exists (shouldn't happen in resume), create new one
  if [ -z "$LOG_FILE" ]; then
    mkdir -p "$SESSIONS_DIR"
    LOG_FILE="$SESSIONS_DIR/$(date +%Y%m%d_%H%M%S).jsonl"
    touch "$LOG_FILE"
  fi
  
  # Run Claude with stream-json output, APPEND to existing log file
  # Terminal that's already watching this file will see new output
  RESULT=$(claude -p "$ANSWER" \
    --dangerously-skip-permissions \
    --continue \
    --setting-sources "" \
    --output-format stream-json \
    --verbose 2>&1 | tee -a "$LOG_FILE" | grep '"type":"result"' | tail -1)
  
  # Extract the actual result text for orchestrator
  if [ -n "$RESULT" ]; then
    echo "$RESULT" | node -e "
      const data = JSON.parse(require('fs').readFileSync('/dev/stdin', 'utf8'));
      if (data.result) console.log(data.result);
    " 2>/dev/null || echo "Task completed. Check log for details."
  else
    echo "Task completed."
  fi
  
  # Extract session_id from stream-json output
  NEW_SESSION_ID=$(grep '"session_id"' "$LOG_FILE" | tail -1 | sed 's/.*"session_id":"\([^"]*\)".*/\1/')
fi

rm -f "$TEMP_OUTPUT"

# Output session_id marker for orchestrator
if [ -n "$NEW_SESSION_ID" ] && [ "$NEW_SESSION_ID" != "null" ]; then
  echo ""
  echo "[SESSION_ID: $NEW_SESSION_ID]"
fi
