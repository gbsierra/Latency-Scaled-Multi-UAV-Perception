# Research Bottleneck + Architecture Choice

## Bottleneck

AirCopBench shows that multi-UAV multimodal models fail in ways that need structured cross-view evidence.

> "The errors in MLLM reasoning primarily stem from three causes: (1) Perception Hallucination Errors... (2) Spatial Reasoning Errors... (3) Multi-image Understanding Errors..."

Source: AirCopBench, arXiv 2511.11025  
Link: https://arxiv.org/abs/2511.11025

## Chosen Architecture

Compare two workflows:

```text
global_single:
all UAV images + question + choices -> one model answer

parallel_uav_fusion:
one observation agent per UAV image, run concurrently -> fusion/comparer agent -> one model answer
```

## Why This Architecture

AirCopBench input is naturally separated by UAV view. A single global call can under-attend to one image or flatten conflicting evidence across images.

The main workflow uses decomposition plus aggregation:

```text
inspect each UAV view independently
preserve a short structured observation per view
compare all observations against the same question and choices
return one final A/B/C/D answer
```

This is not a claim that a second model can reliably verify the first model. A same-context verifier may share the same visual blind spots. The stronger claim is that view-level decomposition changes what evidence reaches the final decision.

## Connection

AirCopBench failure:

```text
hallucination + spatial errors + multi-image errors
```

Per-UAV fusion response:

```text
view-specific observations + cross-view comparison + final answer
```

## What We Test

Not:

```text
this is the best architecture
```

Test:

```text
can parallel per-UAV fusion improve raw accuracy and actionable accuracy over global_single before the decision deadline?
```

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
