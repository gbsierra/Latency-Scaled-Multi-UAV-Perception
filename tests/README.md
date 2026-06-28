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
python3 -m coverage run --source=src -m pytest
python3 -m coverage report --fail-under=80
```

The GitHub check workflow runs the coverage gate on every push and pull request.
