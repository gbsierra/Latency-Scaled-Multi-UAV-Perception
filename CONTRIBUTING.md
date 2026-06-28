# Contributing

GitHub runs the check workflow on every push and pull request.

For local verification before committing, install test dependencies once:

```bash
python3 -m pip install -r requirements-dev.txt
```

Then run the same check command used by the GitHub workflow:

```bash
bash scripts/check
```

Required:

- tests pass
- coverage stays at or above 80%
- tests assert behavior, not imports
