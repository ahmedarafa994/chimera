param (
    [string]$Vendor,
    [string]$Agent,
    [string]$Task
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AgentDir = Join-Path $ScriptDir $Agent
$LogFile = Join-Path $ScriptDir "subagent.log"

Set-Location $AgentDir

if ($Vendor -eq "codex") {
    # CODEX logic
    Add-Content -Path $LogFile -Value "=== [$Agent] START $(Get-Date -Format 'HH:mm:ss') ==="
    
    # Open new PowerShell window to tail log
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Get-Content -Path '$LogFile' -Wait -Tail 200"
    
    # Run Codex
    # Assuming 'codex' is in PATH. If not, needs full path.
    $Env:CODEX_HOME = Join-Path $AgentDir ".codex"
    $Command = "codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox `"First, read ${Agent}.md. Then: $Task`""
    
    # Capture output options need careful handling in PS. 
    # For now, simplistic execution.
    Invoke-Expression $Command | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    # Session ID extraction would go here if needed, but simplified for now.

} else {
    # CLAUDE logic
    $SessionsDir = Join-Path $AgentDir "sessions"
    if (-not (Test-Path $SessionsDir)) {
        New-Item -Path $SessionsDir -ItemType Directory | Out-Null
    }
    
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $SessionLogFile = Join-Path $SessionsDir "${Timestamp}.jsonl"
    $Formatter = Join-Path $ScriptDir "format-log.js"
    
    New-Item -Path $SessionLogFile -ItemType File | Out-Null
    
    # Open tail in new window
    # Paming output through node formatter if desired, or just raw tail
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Get-Content -Path '$SessionLogFile' -Wait -Tail 200"
    
    # Run Claude
    # We use Invoke-Expression or Start-Process. 
    # capturing stdout to log file
    $Prompt = "First, read ${Agent}.agent. Then: $Task"
    
    # Constructing command string for claude
    # Note: quoting can be tricky in PS.
    # We redirect stderr to stdout (2>&1) and tee to file? 
    # PowerShell's Tee-Object is useful.
    
    $ClaudeCmd = "claude -p `"$Prompt`" --dangerously-skip-permissions --output-format stream-json --verbose"
    
    try {
        # Execute and pipe to file. 
        # Using cmd /c to handle the redirection robustly if needed, or PS native
        Invoke-Expression "$ClaudeCmd" | Add-Content -Path $SessionLogFile
        
        # Extract Result
        $Result = Get-Content -Path $SessionLogFile | Select-String -Pattern '"type":"result"' | Select-Object -Last 1
        if ($Result) {
             # Parse JSON roughly or output raw
             # In a real script, we'd use ConvertFrom-Json
             echo "Task completed."
        }
        
        # Extract Session ID
        $SessionIdLine = Get-Content -Path $SessionLogFile | Select-String -Pattern '"session_id"' | Select-Object -First 1
        if ($SessionIdLine -match '"session_id":"([^"]*)"') {
            $SessionId = $Matches[1]
            Write-Output ""
            Write-Output "[SESSION_ID: $SessionId]"
        }
        
    } catch {
        Write-Error "Failed to run Claude: $_"
    }
}
