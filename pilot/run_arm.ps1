# Detached launcher (no session time limit), UTF-8 stdout. Re-running with the same run-id resumes from gepa_state.
param([string]$RunId, [string]$Arm, [string]$BudgetFrom = "")
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$args_ = @("run_optimize.py", "--run-id", $RunId, "--arm", $Arm, "--seed", "0", "--max-metric-calls", "400")
if ($BudgetFrom) { $args_ += @("--budget-from", $BudgetFrom) }
Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" -ArgumentList $args_ -WorkingDirectory $PSScriptRoot `
  -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot\runs\$RunId.log" -RedirectStandardError "$PSScriptRoot\runs\$RunId.err"
