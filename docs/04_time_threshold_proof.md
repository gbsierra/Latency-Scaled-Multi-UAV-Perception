# Time Threshold Proof

## Thesis

Better answers from more agents are only useful if they arrive before the task deadline.

Our metric:

```text
Actionable Accuracy = correct AND latency <= threshold
```

More agents may improve answer quality, but each extra step adds latency. This doc defines task-dependent latency thresholds for the demo, not universal UAV safety limits.

---

## Ordered Latency Ladder

| Latency | Constraint | What it means for us | Source quote |
|---:|---|---|---|
| **100-200 ms** | Human delay perception starts in one mobile-robot teleoperation study | Strong target for real-time feel. Extra agents probably cannot fit unless extremely fast. | “We also identify a threshold window (100-200 ms) for early perception of delay in humans.” [Chen et al., 2025](https://arxiv.org/abs/2508.18074) |
| **150-225 ms** | Closed-loop stability collapsed in one vision-teleoperation lane-keeping study | Hard target for control-like tasks. If we exceed this, do not claim live control. | “Sharp collapse in stability between 150 ms and 225 ms of one-way perception latency.” [Khalil & Kwon, 2026](https://arxiv.org/abs/2603.06850) |
| **200-300 ms** | Human teleoperation performance degrades | Practical warning zone. More agents must clearly improve quality to justify this cost. | “Behavior analysis reveals significant performance degradation at 200-300 ms delays.” [Chen et al., 2025](https://arxiv.org/abs/2508.18074) |
| **400 ms** | Cognitive delay cost saturated in one teleoperation study | Main demo deadline. Better quality after this should be scored stale for this project, not claimed useless in every task. | “When delay exceeds 400 ms, all features plateau, indicating saturation of cognitive resource allocation at physiological limits.” [Chen et al., 2025](https://arxiv.org/abs/2508.18074) |
| **>400 ms** | Stale-decision zone | Use only for comparison. A more accurate answer here may be too late to count. | “Inference latency remains a major obstacle to stable high-frequency control.” [Guo & Guo, 2026](https://arxiv.org/abs/2606.25985) |

---

## How We Use This

Primary target:

```text
stay under the chosen demo deadline, initially 400 ms
```

Stronger target:

```text
stay under 225 ms
```

Main comparison:

```text
Single Agent: lower latency, maybe lower quality
Multi-Agent: higher quality, but only wins if still under threshold
```

A multi-step result only wins under this metric if:

```text
quality improves AND latency <= deadline
```

A result that is better but late is not a win.

---

## Why This Matters

Drone and robot decisions are perishable.

The VLA latency problem is not cosmetic:

> “The next action chunk is still predicted from stale observations while the robot continues to move.” [Guo & Guo, 2026](https://arxiv.org/abs/2606.25985)

Drone LLM planning has the same issue:

> “In real-time and interactive applications involving mobile robots, particularly drones, the sequential token generation process inherent to LLMs introduces substantial latency.” [Chen et al., 2023](https://arxiv.org/abs/2312.14950)

So our thesis is:

```text
Fast inference lets us run per-UAV perception agents and a fusion/comparer without crossing the point where better answers become stale.
```
