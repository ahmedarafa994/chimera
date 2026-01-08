#!/bin/bash
VENDOR="$1"
AGENT="$2"
TASK="$3"

SUBAGENTS_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$SUBAGENTS_DIR/$AGENT"
LOG_FILE="$SUBAGENTS_DIR/subagent.log"
TEMP_OUTPUT=$(mktemp)

cd "$AGENT_DIR"

if [ "$VENDOR" = "codex" ]; then
  # CODEX: stderr contains session_id and verbose logs
  # ISOLATION: CODEX_HOME points to .codex subdir with symlinked auth.json
  # This blocks ~/.codex/AGENTS.md while preserving authentication
  
  # Create/clear log file for new session
  echo "=== [$AGENT] START $(date +%H:%M:%S) ===" > "$LOG_FILE"
  
  # Open Terminal.app with tail -f and bring to front
  osascript -e "tell app \"Terminal\"
    do script \"tail -n 200 -f '$LOG_FILE'\"
    activate
  end tell" &>/dev/null &
  
  # Run Codex with stderr streaming to both log file (real-time) and temp file (for session_id extraction)
  CODEX_HOME="$AGENT_DIR/.codex" codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \
    "First, read ${AGENT}.md. Then: $TASK" 2> >(tee "$TEMP_OUTPUT" >> "$LOG_FILE")
  
  # Extract session_id (strip ANSI codes first)
  SESSION_ID=$(sed 's/\x1b\[[0-9;]*m//g' "$TEMP_OUTPUT" | grep -oE "session id: [0-9a-f-]+" | head -1 | cut -d' ' -f3)
  
else
  # CLAUDE: Real-time formatted JSONL log viewer
  # ISOLATION: --setting-sources "" blocks all CLAUDE.md files
  # Credentials from macOS Keychain remain accessible
  
  # Create sessions directory and log file with known name
  SESSIONS_DIR="$AGENT_DIR/sessions"
  mkdir -p "$SESSIONS_DIR"
  LOG_FILE="$SESSIONS_DIR/$(date +%Y%m%d_%H%M%S).jsonl"
  FORMATTER="$SUBAGENTS_DIR/format-log.js"
  
  # Create empty log file so tail -f starts immediately
  touch "$LOG_FILE"
  
  # Open Terminal.app with tail -f on the known log file
  osascript -e "tell app \"Terminal\"
    do script \"tail -n 200 -f '$LOG_FILE' | node '$FORMATTER'\"
    activate
  end tell" &>/dev/null &
  
  # Small delay to ensure Terminal opens before output starts
  sleep 0.5
  
  # Run Claude with stream-json output, pipe to log file, and parse result for orchestrator
  RESULT=$(claude -p "First, read ${AGENT}.md. Then: $TASK" \
    --dangerously-skip-permissions \
    --setting-sources "" \
    --output-format stream-json \
    --verbose 2>&1 | tee "$LOG_FILE" | grep '"type":"result"' | tail -1)
  
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
  SESSION_ID=$(grep '"session_id"' "$LOG_FILE" | head -1 | sed 's/.*"session_id":"\([^"]*\)".*/\1/')
fi

rm -f "$TEMP_OUTPUT"

# Output session_id marker for orchestrator to parse
if [ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "null" ]; then
  echo ""
  echo "[SESSION_ID: $SESSION_ID]"
fi
