# Contributing

GitHub runs the check workflow on every push and pull request.

For local verification before committing, install test dependencies once:

```bash
python3 -m pip install -r requirements-dev.txt
```

Then run the same coverage gate used by the GitHub workflow:

```bash
python3 -m coverage run --source=src -m pytest
python3 -m coverage report --fail-under=80
```

Required:

- tests pass
- coverage stays at or above 80%
- tests assert behavior, not imports
