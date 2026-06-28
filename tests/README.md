# Tests

```text
tests/
├── unit/         # isolated function/module behavior
└── integration/  # multiple modules with real local data when available
```

Run:

```bash
python3 -m pytest tests/unit
python3 -m pytest tests/integration
bash scripts/check
```

The GitHub check workflow runs `bash scripts/check` on every push and pull request.
