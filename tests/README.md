# Tests

```text
tests/
├── unit/
│   ├── test_aircopbench_cases.py  # raw metadata loader behavior
│   ├── test_benchmark_cases.py    # selected-slice loader and image path resolver
│   ├── test_cerebras_models.py    # Cerebras payload and response parsing
│   ├── test_github_models.py      # GitHub Models payload and response parsing
│   ├── test_image_payloads.py     # local image data-url encoding
│   ├── test_latency_probe.py      # one-case latency probe row construction
│   ├── test_model_clients.py      # provider-neutral request/response boundary
│   ├── test_prompts.py            # multiple-choice prompt formatting
│   ├── test_results.py            # JSONL result persistence
│   ├── test_scoring.py            # answer parsing and deadline scoring
│   ├── test_workflow_runner.py    # workflow runner selection and persistence behavior
│   └── test_workflows.py          # benchmark-compatible workflow orchestration
└── integration/
    └── test_loader_scoring_integration.py  # real local metadata module integration
```

Run:

```bash
python3 -m pytest tests/unit
python3 -m pytest tests/integration
bash scripts/check
```

The GitHub check workflow runs `bash scripts/check` on every push and pull request.
