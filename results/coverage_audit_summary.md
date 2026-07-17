# Coverage Audit Summary

- Scope: Person 3 Random Forest files
- Timestamp: 2026-07-15T17:39:42
- Command: `C:\Users\nihat\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe -m pytest tests/test_random_forest.py tests/test_rf_utils.py tests/test_rf_experiments.py --cov=src.bagging.random_forest --cov=src.experiments.rf_utils --cov=src.experiments.rf_scaling --cov=src.experiments.noise_robustness --cov=src.experiments.rf_parallel_benchmark --cov-report=term-missing`
- Exit code: `0`
- Total coverage: **74%**
- Raw log: `results/coverage_audit_raw.log`

The project-wide minimum is 60%. Person 3 must report modules below
the threshold to their owners before the final integration PR.
