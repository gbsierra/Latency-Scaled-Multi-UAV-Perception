# Correct but late is stale.

AirCopBench shows multi-UAV perception is hard.  
This project asks the next question:

> Can parallel per-UAV perception agents and one fusion agent make a better multi-UAV decision before the deadline?

The point is not just faster inference.

The point is:

> fast inference may make a better workflow possible before time runs out.

## What this demo tests

Two workflows:

```text
global_single:
all UAV images + question + choices -> one model answer

parallel_uav_fusion:
one concurrent observation per UAV image -> fusion/comparer -> one model answer
```

Scored by:

```text
actionable = correct AND latency_ms <= deadline_ms
```

## Start here

1. [Product Definition](docs/01_product_definition.md)  
   What the demo is, what it shows, and what it is not.

2. [Research Bottleneck + Architecture Choice](docs/02_research_bottleneck_architecture.md)  
   Why AirCopBench failure modes justify per-UAV perception plus fusion.

3. [AirCopBench Paper Notes](docs/03_aircopbench_paper_notes.md)  
   What the benchmark provides and why it is a good testbed.

4. [Time Threshold Proof](docs/04_time_threshold_proof.md)  
   Why better answers only matter if they arrive before the deadline.

## What this is not

- not drone control
- not a full benchmark leaderboard
- not a fake simulator
- not mocked results
- not a claim that more agents always win

## Core claim

AirCopBench measures correctness.  
This project measures whether a multi-UAV workflow returns a correct answer in time to act.

## Citation

This project uses AirCopBench as the multi-UAV benchmark source.

```bibtex
@misc{zha2025aircopbench,
  title={AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning},
  author={Zha, Jirong and Fan, Yuxuan and Zhang, Tianyu and Chen, Geng and Chen, Yingfeng and Gao, Chen and Chen, Xinlei},
  year={2025},
  eprint={2511.11025},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.11025}
}
```
