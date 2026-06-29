# Correct but late is stale.

[AirCopBench](https://arxiv.org/abs/2511.11025) shows multi-UAV perception is hard.  
Cerebras-speed inference raises the next applied question:

> Can faster multimodal inference create room for richer multi-UAV workflows that improve hard benchmark answers before they become stale?

This project tests one research-motivated idea: parallel per-UAV perception followed by fusion of the results.

The point is:

> lower latency expands the design space for applied UAV workflows, but adding more inference calls only helps when the workflow improves the right cases enough to justify its latency cost.
## What this demo tests

![Global vs per-UAV fusion execution flow](docs/assets/01_global_vs_fusion_execution_flow.png)

The comparison is simple:

- `global_single`: one model call over all UAV images.
- `parallel_uav_fusion`: one model call per UAV image, then one final call to combine the notes.

Scored by:

```text
actionable = correct AND latency_ms <= deadline_ms
```

## What we found

`global_single` was the stronger default workflow, while `parallel_uav_fusion` fixed some cases but also introduced comparable regressions and added latency.

**See the full benchmark story here:** [Benchmark Results](docs/05_benchmark_results.md)


## Start here

Note: For the original benchmark task definitions, see: [AirCopBench task definitions](https://github.com/zhajirong/AirCopBench#task-definition)

1. [Product Definition](docs/01_product_definition.md)  
   What the demo is, what it shows, and what it is not.

2. [Research Bottleneck + Architecture Choice](docs/02_research_bottleneck_architecture.md)  
   Why AirCopBench failure modes justify per-UAV perception plus fusion.

3. [AirCopBench Paper Notes](docs/03_aircopbench_paper_notes.md)  
   What the benchmark provides and why it is a good testbed.

4. [Time Threshold Proof](docs/04_time_threshold_proof.md)  
   Why better answers only matter if they arrive before the deadline.

## Citation

This project uses AirCopBench as the multi-UAV benchmark source.

```bibtex
@inproceedings{zha2026aircopbench,
  title={Aircopbench: A benchmark for multi-drone collaborative embodied perception and reasoning},
  author={Zha, Jirong and Fan, Yuxuan and Zhang, Tianyu and Chen, Geng and Chen, Yingfeng and Gao, Chen and Chen, Xinlei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={2},
  pages={1507--1515},
  year={2026}
}
```
