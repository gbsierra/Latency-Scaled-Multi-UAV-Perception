# Agent Notes

Project thesis:

- core thesis: correct but late is stale
- product claim: measure which multimodal UAV workflow produces actionable decisions under a deadline
- technical claim: fast inference can make `parallel per-UAV perception -> fusion/comparer` feasible inside the decision window

Read first:

- `README.md` for the project thesis
- `CONTRIBUTING.md` before code changes


Use `docs/` for public project docs. Use `docs/ignore/` only for local planning notes.

Code rules:

- keep changes small and durable
- do not create transformed datasets unless needed
- preserve AirCopBench case fields
- add generic functions before workflow-specific wrappers
- add short comments for intent or non-obvious logic only
- update `src/README.md` or `tests/README.md` when adding files there

Testing rules:

- add tests when changing `src/`
- put unit tests in `tests/unit/`
- put integration tests in `tests/integration/`
- split tests by source file when practical
- use unit tests for function behavior
- use integration tests when modules must work together
- skip integration tests cleanly when gitignored raw data is missing
- tests must assert behavior, not imports
- coverage must stay at or above 80%

Run locally before handing off code changes:

```bash
bash scripts/check
```

GitHub also runs this command through the check workflow on every push and pull request.
