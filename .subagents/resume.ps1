param (
    [string]$Vendor,
    [string]$Agent,
    [string]$SessionId,
    [string]$Answer
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AgentDir = Join-Path $ScriptDir $Agent
$LogFile = Join-Path $ScriptDir "subagent.log"

Set-Location $AgentDir

if ($Vendor -eq "codex") {
    # CODEX
    Add-Content -Path $LogFile -Value "=== [$Agent] RESUME $(Get-Date -Format 'HH:mm:ss') ==="
    
    $Env:CODEX_HOME = Join-Path $AgentDir ".codex"
    $Command = "codex exec --dangerously-bypass-approvals-and-sandbox resume `"$SessionId`" `"$Answer`""
    
    Invoke-Expression $Command | Out-File -FilePath $LogFile -Append -Encoding utf8

} else {
    # CLAUDE
    $SessionsDir = Join-Path $AgentDir "sessions"
    
    # Find latest jsonl
    $SessionLogFile = Get-ChildItem -Path $SessionsDir -Filter "*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    
    if (-not $SessionLogFile) {
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $SessionLogFile = Join-Path $SessionsDir "${Timestamp}.jsonl"
        New-Item -Path $SessionLogFile -ItemType File | Out-Null
    }
    
    # Run Claude continue
    $ClaudeCmd = "claude -p `"$Answer`" --dangerously-skip-permissions --continue --setting-sources `"`" --output-format stream-json --verbose"
    
    try {
        Invoke-Expression "$ClaudeCmd" | Add-Content -Path $SessionLogFile
        
        # Extract Session ID (new one if rotated?)
        $SessionIdLine = Get-Content -Path $SessionLogFile | Select-String -Pattern '"session_id"' | Select-Object -Last 1
        if ($SessionIdLine -match '"session_id":"([^"]*)"') {
            $NewSessionId = $Matches[1]
            Write-Output ""
            Write-Output "[SESSION_ID: $NewSessionId]"
        }
        
    } catch {
        Write-Error "Failed to run Claude: $_"
    }
}
