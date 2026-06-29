"""Streamlit replay app for saved AirCopBench workflow benchmark runs."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from html import escape

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

# Saved benchmark artifacts keep the demo deterministic for recorded presentations.
V1_CASES_PATH = ROOT / "data/selected/aircopbenchv1/selection/object_matching_86_cases.json"
V1_DATASET_ROOT = ROOT / "data/selected/aircopbenchv1"
V1_RESULTS_PATH = ROOT / "results/benchmark_v1/object_matching_86.jsonl"

V0_CASES_PATH = ROOT / "data/selected/aircopbench/selection/core_30_cases.json"
V0_DATASET_ROOT = ROOT / "data/selected/aircopbench"
V0_CEREBRAS_RESULTS_PATH = ROOT / "results/benchmark/benchmark_v0.jsonl"
V0_PHI_RESULTS_PATH = ROOT / "results/benchmark/benchmark_v0_phi.jsonl"

GLOBAL_WORKFLOW = "global_single"
FUSION_WORKFLOW = "parallel_uav_fusion"

# Hand-picked replay cases for the three demo beats.
GLOBAL_WIN_CASE_ID = "Sim3_OM_UAV2_561"
FUSION_WIN_CASE_ID = "MDMT_OM_UAV5_27"
SPEED_CASE_ID = "Sim5_what2col_UAV4_1"
SPEED_WORKFLOW = FUSION_WORKFLOW


st.set_page_config(
    page_title="AirCopBench Workflow Replay",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def read_json(path: str) -> Any:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["question_id"]): case for case in cases}


def workflow_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    # Key by case and workflow so each page can pull matched rows quickly.
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            continue
        indexed[str(row["question_id"])][str(row["workflow"])] = row
    return indexed


def provider_workflow_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        indexed[(str(row["provider"]), str(row["question_id"]), str(row["workflow"]))] = row
    return indexed


def seconds(ms: int | float | None) -> str:
    if not isinstance(ms, int | float):
        return "n/a"
    return f"{ms / 1000:.2f}s"


def answer_label(row: dict[str, Any]) -> str:
    parsed = row.get("parsed_answer")
    if not parsed:
        return "unparsed"
    return str(parsed)


def result_state(row: dict[str, Any]) -> str:
    return "correct" if row.get("correct") is True else "wrong"


def result_color(row: dict[str, Any]) -> str:
    return "#079455" if row.get("correct") is True else "#D92D20"


def inject_css() -> None:
    # Streamlit is used as a shell; the demo surface is mostly custom HTML/CSS.
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; }
        h1 { margin-bottom: 0.25rem; }
        h2 { margin-top: 0.45rem; margin-bottom: 0.3rem; }
        h3 { margin-top: 0.35rem; margin-bottom: 0.35rem; }
        div[data-testid="stVerticalBlock"] { gap: 0.45rem; }
        .demo-kicker {
            color: #475467;
            font-size: 0.92rem;
            line-height: 1.25rem;
            margin: 0;
        }
        .demo-heading {
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
            margin: 0.45rem 0 0.25rem 0;
        }
        .demo-heading-title {
            color: #172026;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 2rem;
        }
        .demo-heading-meta {
            color: #667085;
            font-size: 0.88rem;
            line-height: 1.1rem;
        }
        .timer-box {
            border: 1px solid #D0D5DD;
            border-radius: 8px;
            padding: 8px 12px;
            background: #F8FAFC;
        }
        .timer-label {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .timer-value {
            color: #172026;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.9rem;
            font-variant-numeric: tabular-nums;
        }
        .uav-strip img {
            max-height: 145px;
            object-fit: cover;
        }
        .choice-box {
            border: 1px solid #EAECF0;
            border-radius: 8px;
            padding: 6px 9px;
            margin-bottom: 5px;
            background: #FFFFFF;
            color: #172026;
        }
        .choice-letter {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            margin-right: 7px;
            border: 2px solid #98A2B3;
            border-radius: 999px;
            font-weight: 750;
            color: #172026;
        }
        .choice-text {
            font-size: 0.84rem;
            line-height: 1.08rem;
        }
        .choice-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 5px;
            margin-left: 34px;
        }
        .answer-pill {
            display: inline-block;
            border-left: 4px solid;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 0.78rem;
            font-weight: 700;
            background: #F8FAFC;
            line-height: 1.05rem;
        }
        .choices-spacer {
            height: 8px;
        }
        .uav-caption {
            color: #344054;
            font-weight: 700;
            font-size: 0.8rem;
            margin: 0 0 3px 0;
        }
        .trace-panel {
            border: 1px solid #D0D5DD;
            border-radius: 8px;
            padding: 9px 11px;
            background: #FFFFFF;
            min-height: 260px;
        }
        .trace-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 7px;
        }
        .trace-title {
            color: #172026;
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.15rem;
        }
        .trace-status {
            border-radius: 999px;
            padding: 2px 8px;
            color: #FFFFFF;
            font-size: 0.76rem;
            font-weight: 800;
            line-height: 1.05rem;
        }
        .trace-event {
            border-left: 4px solid #98A2B3;
            padding: 5px 8px;
            margin: 5px 0;
            background: #F8FAFC;
            border-radius: 4px;
            color: #172026;
            font-size: 0.86rem;
            line-height: 1.12rem;
        }
        .trace-details {
            border-left: 4px solid #98A2B3;
            margin: 5px 0;
            background: #F8FAFC;
            border-radius: 4px;
            color: #172026;
            font-size: 0.86rem;
            line-height: 1.12rem;
        }
        .trace-details summary {
            cursor: pointer;
            padding: 5px 8px;
            list-style-position: inside;
        }
        .trace-raw {
            margin: 0;
            padding: 8px 10px 10px 14px;
            border-top: 1px solid #EAECF0;
            white-space: normal;
            color: #344054;
            font-size: 0.78rem;
            line-height: 1.05rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }
        .trace-time {
            display: inline-block;
            color: #667085;
            font-weight: 800;
            min-width: 3.7rem;
            font-variant-numeric: tabular-nums;
        }
        .trace-muted {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.1rem;
            margin-top: 5px;
        }
        .verdict-box {
            border-radius: 8px;
            padding: 8px 11px;
            background: #FFF4E5;
            color: #93370D;
            border: 1px solid #FDBA74;
            font-weight: 750;
        }
        .small-muted { color: #667085; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_question_and_choices(
    case: dict[str, Any],
    annotations: list[dict[str, str]],
) -> None:
    st.subheader("Question")
    st.write(case["question"])
    render_answer_choices(case, annotations)


def render_answer_choices(
    case: dict[str, Any],
    annotations: list[dict[str, str]],
) -> None:
    # Answer markers appear only after the corresponding trace has finished.
    st.subheader("Answer choices")
    for letter in ("A", "B", "C", "D"):
        pills = []
        for annotation in annotations:
            if annotation["answer"] != letter:
                continue
            color = escape(annotation["color"])
            label = escape(annotation["label"])
            pills.append(f'<span class="answer-pill" style="border-left-color:{color}; color:{color};">{label}</span>')
        pill_html = "".join(pills)
        choice_text = escape(str(case["options"][letter]))
        st.markdown(
            (
                '<div class="choice-box">'
                f'<div><span class="choice-letter">{letter}</span>'
                f'<span class="choice-text">{choice_text}</span></div>'
                f'<div class="choice-pills">{pill_html}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
    st.markdown('<div class="choices-spacer"></div>', unsafe_allow_html=True)


def render_compact_images(case: dict[str, Any], dataset_root: Path) -> None:
    st.markdown('<div class="uav-strip">', unsafe_allow_html=True)
    image_items = list(case["uav_paths"].items())
    columns = st.columns(len(image_items), gap="small")
    for column, (uav_id, relative_path) in zip(columns, image_items, strict=True):
        image_path = dataset_root / relative_path
        with column:
            st.markdown(f'<div class="uav-caption">{escape(str(uav_id))}</div>', unsafe_allow_html=True)
            st.image(str(image_path), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def event_time(ms: int | float | None) -> str:
    if not isinstance(ms, int | float):
        return "--.--s"
    return f"{ms / 1000:05.2f}s"


def observer_support(raw_observation: str) -> str:
    # Observer prompts were structured, so the support line is useful trace text.
    match = re.search(r"supports:\s*([^\n]+)", raw_observation, flags=re.IGNORECASE)
    if not match:
        return "support unknown"
    return f"supports {match.group(1).strip()}"


def observer_evidence(raw_observation: str) -> str:
    for line in raw_observation.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            return stripped.removeprefix("-").strip()
    return raw_observation.strip().splitlines()[0] if raw_observation.strip() else "observation saved"


def trace_status(row: dict[str, Any], finished: bool) -> tuple[str, str]:
    if not finished:
        return "running", "#475467"
    if row.get("correct") is True:
        return "correct", "#079455"
    return "wrong", "#D92D20"


def render_trace_panel(
    title: str,
    row: dict[str, Any],
    events: list[dict[str, Any]],
    elapsed_ms: float,
    accent: str,
) -> None:
    finished = elapsed_ms >= float(row.get("latency_ms", 0))
    status, color = trace_status(row, finished)
    event_html = ""
    for event in events:
        # Replay saved events according to recorded wall-clock latency.
        if elapsed_ms < float(event["time_ms"]):
            continue
        label = (
            f'<span class="trace-time">{event_time(event["time_ms"])}</span>'
            f'{escape(str(event["label"]))}'
        )
        if event.get("raw"):
            # Native details keeps raw observer output attached to its trace row.
            raw_html = escape(str(event["raw"]).strip()).replace("\n", "<br>")
            event_html += (
                f'<details class="trace-details" style="border-left-color:{accent};">'
                f"<summary>{label}</summary>"
                f'<div class="trace-raw">{raw_html}</div>'
                "</details>"
            )
        else:
            event_html += (
                f'<div class="trace-event" style="border-left-color:{accent};">'
                f"{label}"
                '</div>'
            )

    if not finished:
        event_html += '<div class="trace-muted">Waiting for saved benchmark events...</div>'

    st.markdown(
        (
            '<div class="trace-panel">'
            '<div class="trace-head">'
            f'<div class="trace-title">{escape(title)}</div>'
            f'<div class="trace-status" style="background:{color};">{escape(status)}</div>'
            '</div>'
            f"{event_html}"
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def global_trace_events(global_row: dict[str, Any]) -> list[dict[str, Any]]:
    # The final global model response was constrained to one answer letter.
    return [
        {"time_ms": 0, "label": "global call started with all UAV images"},
        {
            "time_ms": global_row["latency_ms"],
            "label": (
                f"final answer {answer_label(global_row)} "
                f"({result_state(global_row)}, {seconds(global_row.get('latency_ms'))})"
            ),
        },
    ]


def fusion_trace_events(fusion_row: dict[str, Any]) -> list[dict[str, Any]]:
    # Fusion traces expose observer completion before the final comparer call.
    events: list[dict[str, Any]] = [{"time_ms": 0, "label": "per-UAV observers started in parallel"}]
    observers = fusion_row.get("observer_outputs")
    if isinstance(observers, list):
        for observer in sorted(observers, key=lambda item: float(item.get("latency_ms", 0))):
            raw = str(observer.get("raw_observation", "")).strip()
            events.append(
                {
                    "time_ms": observer.get("latency_ms", 0),
                    "uav_id": observer.get("uav_id", "UAV"),
                    "label": (
                        f"{observer.get('uav_id', 'UAV')} observer ready: "
                        f"{observer_support(raw)}; {observer_evidence(raw)}"
                    ),
                    "raw": raw,
                }
            )

    observer_max = fusion_row.get("observer_max_latency_ms")
    if isinstance(observer_max, int | float):
        events.append({"time_ms": observer_max, "label": "fusion/comparer started from observer notes"})
    events.append(
        {
            "time_ms": fusion_row["latency_ms"],
            "label": (
                f"final answer {answer_label(fusion_row)} "
                f"({result_state(fusion_row)}, {seconds(fusion_row.get('latency_ms'))})"
            ),
        }
    )
    return sorted(events, key=lambda event: float(event["time_ms"]))


def annotations_for_trace_specs(
    specs: list[dict[str, Any]],
    gold_answer: str,
    elapsed_ms: float,
) -> list[dict[str, str]]:
    annotations = [{"answer": gold_answer, "label": "ground truth", "color": "#172026"}]
    for spec in specs:
        row = spec["row"]
        if elapsed_ms < float(row["latency_ms"]):
            continue
        annotations.append(
            {
                "answer": answer_label(row),
                "label": f"{spec['label']} · {seconds(row.get('latency_ms'))} · {result_state(row)}",
                "color": spec["color"],
            }
        )
    return annotations


def render_timer(elapsed_ms: float) -> None:
    st.markdown(
        (
            '<div class="timer-box">'
            '<div class="timer-label">Replay timer</div>'
            f'<div class="timer-value">{elapsed_ms / 1000:04.1f}s</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_demo_heading(title: str, case_id: str, case: dict[str, Any]) -> None:
    st.markdown(
        (
            '<div class="demo-heading">'
            f'<span class="demo-heading-title">{escape(title)}</span>'
            f'<span class="demo-heading-meta">Case {escape(case_id)} · {escape(str(case.get("source_group")))}</span>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_timed_trace_page(
    title: str,
    case_id: str,
    case: dict[str, Any],
    dataset_root: Path,
    left_title: str,
    left_row: dict[str, Any],
    left_events: list[dict[str, Any]],
    left_color: str,
    right_title: str,
    right_row: dict[str, Any],
    right_events: list[dict[str, Any]],
    right_color: str,
    annotation_specs: list[dict[str, Any]],
    conclusion: str,
    button_key: str,
) -> None:
    # Shared layout for all three demo pages.
    max_latency_ms = max(float(left_row["latency_ms"]), float(right_row["latency_ms"]))

    title_col, timer_col, button_col = st.columns([0.62, 0.18, 0.2], gap="large")
    with title_col:
        render_demo_heading(title, case_id, case)
    with timer_col:
        timer_placeholder = st.empty()
    with button_col:
        begin = st.button("Begin", key=button_key, width="stretch")

    media_col, qa_col = st.columns([0.55, 0.45], gap="large")
    with media_col:
        render_compact_images(case, dataset_root)
    with qa_col:
        st.markdown("**Question**")
        st.markdown(f'<div class="demo-kicker">{escape(str(case["question"]))}</div>', unsafe_allow_html=True)
        choices_placeholder = st.empty()

    global_col, fusion_col = st.columns(2, gap="large")
    with global_col:
        global_placeholder = st.empty()
    with fusion_col:
        fusion_placeholder = st.empty()
    verdict_placeholder = st.empty()

    def render_frame(elapsed_ms: float) -> None:
        # Each frame redraws the same placeholders so the page does not grow.
        with timer_placeholder:
            render_timer(elapsed_ms)
        with choices_placeholder.container():
            render_answer_choices(
                case,
                annotations_for_trace_specs(annotation_specs, str(case["correct_answer"]), elapsed_ms),
            )
        with global_placeholder.container():
            render_trace_panel(
                left_title,
                left_row,
                left_events,
                elapsed_ms,
                left_color,
            )
        with fusion_placeholder.container():
            render_trace_panel(
                right_title,
                right_row,
                right_events,
                elapsed_ms,
                right_color,
            )
        with verdict_placeholder:
            if elapsed_ms >= max_latency_ms:
                st.markdown(
                    (
                        '<div class="verdict-box">'
                        f"{escape(conclusion)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    if begin:
        # Replay is local and deterministic; no provider calls happen here.
        started_at = time.perf_counter()
        while True:
            elapsed_ms = min((time.perf_counter() - started_at) * 1000, max_latency_ms)
            render_frame(elapsed_ms)
            if elapsed_ms >= max_latency_ms:
                break
            time.sleep(0.12)
        render_frame(max_latency_ms)
    else:
        render_frame(0.0)


def render_global_wins_trace(
    cases: dict[str, dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    dataset_root: Path,
) -> None:
    case = cases[GLOBAL_WIN_CASE_ID]
    global_row = results[GLOBAL_WIN_CASE_ID][GLOBAL_WORKFLOW]
    fusion_row = results[GLOBAL_WIN_CASE_ID][FUSION_WORKFLOW]
    render_timed_trace_page(
        title="Global single-call wins",
        case_id=GLOBAL_WIN_CASE_ID,
        case=case,
        dataset_root=dataset_root,
        left_title="global_single",
        left_row=global_row,
        left_events=global_trace_events(global_row),
        left_color="#2563EB",
        right_title="parallel_uav_fusion",
        right_row=fusion_row,
        right_events=fusion_trace_events(fusion_row),
        right_color="#DC6803",
        annotation_specs=[
            {"row": global_row, "label": "global_single", "color": result_color(global_row)},
            {"row": fusion_row, "label": "parallel_fusion", "color": result_color(fusion_row)},
        ],
        conclusion=(
            "Conclusion: Fusion regressed because the observers disagreed. UAV1 supported the correct answer, "
            "UAV3 supported the wrong answer, and the fusion call chose UAV3."
        ),
        button_key="begin_global_wins",
    )


def render_fusion_wins_trace(
    cases: dict[str, dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    dataset_root: Path,
) -> None:
    case = cases[FUSION_WIN_CASE_ID]
    global_row = results[FUSION_WIN_CASE_ID][GLOBAL_WORKFLOW]
    fusion_row = results[FUSION_WIN_CASE_ID][FUSION_WORKFLOW]
    render_timed_trace_page(
        title="Parallel fusion wins",
        case_id=FUSION_WIN_CASE_ID,
        case=case,
        dataset_root=dataset_root,
        left_title="global_single",
        left_row=global_row,
        left_events=global_trace_events(global_row),
        left_color="#2563EB",
        right_title="parallel_uav_fusion",
        right_row=fusion_row,
        right_events=fusion_trace_events(fusion_row),
        right_color="#DC6803",
        annotation_specs=[
            {"row": global_row, "label": "global_single", "color": result_color(global_row)},
            {"row": fusion_row, "label": "parallel_fusion", "color": result_color(fusion_row)},
        ],
        conclusion=(
            "Conclusion: Fusion recovered the correct answer. The global call chose A, while the observer traces "
            "surfaced that UAV1 directly supported the correct answer D."
        ),
        button_key="begin_fusion_wins",
    )


def render_cerebras_faster_trace(
    cases: dict[str, dict[str, Any]],
    cerebras_rows: dict[tuple[str, str, str], dict[str, Any]],
    phi_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    case = cases[SPEED_CASE_ID]
    cerebras = cerebras_rows[("cerebras", SPEED_CASE_ID, SPEED_WORKFLOW)]
    phi = phi_rows[("github_models", SPEED_CASE_ID, SPEED_WORKFLOW)]
    render_timed_trace_page(
        title="Cerebras is faster",
        case_id=SPEED_CASE_ID,
        case=case,
        dataset_root=V0_DATASET_ROOT,
        left_title="Cerebras / Gemma 4 fusion",
        left_row=cerebras,
        left_events=fusion_trace_events(cerebras),
        left_color="#2563EB",
        right_title="GitHub Models / Phi-4 fusion",
        right_row=phi,
        right_events=fusion_trace_events(phi),
        right_color="#7C3AED",
        annotation_specs=[
            {"row": cerebras, "label": "Cerebras/Gemma", "color": "#2563EB"},
            {"row": phi, "label": "Phi/GitHub", "color": "#7C3AED"},
        ],
        conclusion=(
            f"Conclusion: Same case, same fusion workflow, same answer. Cerebras finished in {seconds(cerebras.get('latency_ms'))}; "
            f"Phi finished in {seconds(phi.get('latency_ms'))}."
        ),
        button_key="begin_cerebras_faster",
    )


def main() -> None:
    inject_css()

    # Load once per Streamlit cache; all pages replay from these saved rows.
    v1_cases = case_index(read_json(str(V1_CASES_PATH)))
    v1_results = workflow_index(read_jsonl(str(V1_RESULTS_PATH)))
    v0_cases = case_index(read_json(str(V0_CASES_PATH)))
    v0_cerebras_rows = provider_workflow_index(read_jsonl(str(V0_CEREBRAS_RESULTS_PATH)))
    v0_phi_rows = provider_workflow_index(read_jsonl(str(V0_PHI_RESULTS_PATH)))

    page = st.radio(
        "Demo page",
        ["Global wins", "Fusion wins", "Cerebras faster"],
        horizontal=True,
    )
    if page == "Global wins":
        render_global_wins_trace(v1_cases, v1_results, V1_DATASET_ROOT)
    elif page == "Fusion wins":
        render_fusion_wins_trace(v1_cases, v1_results, V1_DATASET_ROOT)
    else:
        render_cerebras_faster_trace(v0_cases, v0_cerebras_rows, v0_phi_rows)


if __name__ == "__main__":
    main()
