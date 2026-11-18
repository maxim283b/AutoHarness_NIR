# Harness optimization experiment

Из корня репозитория, PowerShell:

```powershell
New-Item -ItemType Directory -Force .test_tmp | Out-Null
$env:TEMP=(Resolve-Path '.\\.test_tmp').Path
$env:TMP=$env:TEMP
$env:PYTHONPATH=(Resolve-Path '.').Path

# Проверка LM Studio
Invoke-RestMethod http://127.0.0.1:1234/v1/models

# Тесты
& '.\\pilot\\.venv\\Scripts\\python.exe' -m pytest experiments\\harness_optimization\\tests -q --basetemp .test_tmp\\harness_pytest

# Smoke всех методов; повтор той же команды безопасно продолжает run
& '.\\pilot\\.venv\\Scripts\\python.exe' -m experiments.harness_optimization.cli smoke --run-id smoke_20261009

# Симметричный complete_core; повтор команды выполняет resume
& '.\\pilot\\.venv\\Scripts\\python.exe' -m experiments.harness_optimization.cli run --run-id core_20261009

# Полный пересчёт summary и audit только из raw JSONL
& '.\\pilot\\.venv\\Scripts\\python.exe' -m experiments.harness_optimization.cli audit --run-dir '.\\results\\harness_optimization\\core_20261009'
```

Held-out этими командами не открывается.

