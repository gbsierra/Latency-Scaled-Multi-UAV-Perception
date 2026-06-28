# Source

```text
src/
├── aircopbench_cases.py  # load and validate AirCopBench metadata
├── benchmark_cases.py    # load selected benchmark cases and resolve image paths
├── cerebras_models.py    # call Cerebras with local image payloads
├── github_models.py      # call GitHub Models with local image payloads
├── image_payloads.py     # encode local images for multimodal provider payloads
├── latency_probe.py      # run one-case provider latency probes
├── model_clients.py      # normalize provider requests, responses, and latency
├── prompts.py            # build multiple-choice prompts
├── results.py            # persist scored result rows
└── scoring.py            # parse answers and score deadline metrics
```
