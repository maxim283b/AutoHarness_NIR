# Final harness experiment

Gate A1 (`official_infrastructure_reproduction_passed`) и Gate A2
(`real_target_benchmark_integration_passed`) завершены. A1 является только fake
infrastructure fixture; A2 — реальный SQL/LM Studio integration smoke. Ни один из них
не является optimizer screening или validation.

```powershell
$env:PYTHONPATH='experiments\final_harness_experiment\windows_compat;third_party\AutoSaddler\src;.'
$py = '.\pilot\.venv\Scripts\python.exe'
& $py -m pytest experiments\final_harness_experiment\tests -q
& $py -m experiments.final_harness_experiment.gate_a2_audit results\final_harness_experiment\gate_a2_sql\gate-a2-sql-v1
```

Исходная preregistration не изменяется. Staged amendment и новый scientific protocol
должны использоваться для последующих screening/confirmation runs.
