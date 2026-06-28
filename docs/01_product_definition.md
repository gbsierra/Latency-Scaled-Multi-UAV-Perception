# Product Definition

## Product

A small demo that shows a time-sensitive multimodal drone decision.

It uses real multi-UAV benchmark cases.

It shows whether parallel per-UAV perception plus fusion can improve the answer before the deadline.

## Core Idea

Global single-call answers may be weak or wrong.

One global model call can under-attend to one or more UAV views.

Per-UAV perception agents force each view to be inspected independently, then a fusion/comparer agent consolidates the observations into one answer.

But the extra workflow only matters if it finishes in time.

## What The Demo Should Show

One clear case where possible:

- multi-UAV images
- question
- answer choices
- ground truth
- global_single answer
- parallel per-UAV fusion answer
- latency
- deadline status

## What Users Should Understand

The point is not just speed.

The point is:

```text
fast inference can make a better workflow possible
```

## Demo Case Selection

Pick a case only after testing.

Good signs:

- easy to explain visually
- ground truth is clear
- global_single answer is weak or wrong
- per-UAV fusion helps or changes the decision
- latency difference matters
- result supports the main claim

## Questions To Answer Later

- Which case best shows the issue?
- Does per-UAV fusion improve raw accuracy or actionable accuracy?
- What deadline is fair?
- Is a real baseline available?
- What metric is strongest?
- Is the demo clear without extra explanation?

## What This Is Not

- not a full benchmark product
- not a drone simulator
- not real drone control
- not fake autonomy
- not mocked results

## Rule

Use only real cases, real outputs, real labels, and real latency.
