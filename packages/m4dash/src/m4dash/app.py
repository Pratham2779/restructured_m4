"""M4 Results Dashboard: Streamlit Application (Pure Read-Only Consumer)."""
import os
import streamlit as st
import pandas as pd
from m4dash.utils import discover_runs, load_run_data, load_item_trace, create_run_zip

st.set_page_config(
    page_title="M4 Results Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_metric_pill(val, exp_val):
    if isinstance(exp_val, (int, float)):
        diff = val - exp_val
        color = "green" if diff >= 0 else "red"
        symbol = "↑" if diff >= 0 else "↓"
        return f"{val:.2f} (:{color}[{symbol} {abs(diff):.2f} vs {exp_val}])"
    return f"{val:.2f} (:gray[[{exp_val}]])"


def main():
    st.title("🧠 M4 Benchmark Results Dashboard")
    st.caption("Meta-Memory for Large Language Models (Liang et al., IEEE TASLP 2026)")

    # 1. Sidebar - Run Discovery & Selection
    st.sidebar.header("📁 Benchmark Runs")
    base_results_dir = os.environ.get("RESULTS_DIR", "results")
    runs = discover_runs(base_results_dir)

    if not runs:
        st.warning(f"No benchmark runs found under `{base_results_dir}`.")
        st.info(
            "To generate evaluation results, run:\n"
            "```bash\n"
            "m4bench run locomo --stage smoke\n"
            "# or\n"
            "m4bench run msc --stage smoke\n"
            "```"
        )
        return

    run_labels = [f"[{r['dataset'].upper()}] {r['run_id']} ({r['stage']})" for r in runs]
    selected_idx = st.sidebar.selectbox("Select Run", range(len(runs)), format_func=lambda i: run_labels[i])
    selected_run = runs[selected_idx]
    run_dir = selected_run["run_dir"]

    data = load_run_data(run_dir)
    manifest = data["manifest"]
    aggregate = data["aggregate"]
    items = data["items"]
    expected = data["expected"]

    # Sidebar Run Info
    st.sidebar.divider()
    st.sidebar.markdown(f"**Dataset:** `{manifest.get('dataset', selected_run['dataset'])}`")
    st.sidebar.markdown(f"**Stage:** `{manifest.get('stage', selected_run['stage'])}`")
    st.sidebar.markdown(f"**Model:** `{manifest.get('model_name', 'N/A')}`")
    st.sidebar.markdown(f"**Embedder:** `{manifest.get('embedder_name', 'N/A')}`")
    st.sidebar.markdown(f"**Threshold (γ):** `{manifest.get('gamma', 'N/A')}`")
    st.sidebar.markdown(f"**Commit:** `{manifest.get('git_sha', 'unknown')[:7]}`")
    st.sidebar.markdown(f"**Start:** `{manifest.get('start_time', 'N/A')}`")
    if manifest.get("end_time"):
        st.sidebar.markdown(f"**End:** `{manifest.get('end_time')}`")

    # Export ZIP button
    zip_bytes = create_run_zip(run_dir)
    st.sidebar.download_button(
        label="📦 Download Run as ZIP",
        data=zip_bytes,
        file_name=f"{selected_run['dataset']}_{selected_run['run_id']}.zip",
        mime="application/zip",
    )

    # 2. Known Protocol Deviations Banner
    deviations = aggregate.get("known_deviations") or manifest.get("known_deviations", [])
    if deviations:
        with st.expander("⚠️ Known Protocol Deviations vs. Original Paper", expanded=True):
            for d in deviations:
                st.markdown(f"- {d}")

    # 3. Main Dashboard Tabs
    tab_metrics, tab_items, tab_trace, tab_raw = st.tabs([
        "📊 Metrics & Aggregates",
        "🔍 Item Inspector",
        "🕸️ Retrieval Trace Viewer",
        "📄 Raw JSON Artifacts",
    ])

    # --- TAB 1: Metrics & Aggregates ---
    with tab_metrics:
        # High-level KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Items", aggregate.get("total_items", len(items)))
        kpi2.metric("Avg Latency", f"{aggregate.get('avg_latency_seconds', 0.0):.2f}s")
        kpi3.metric("LLM Calls", aggregate.get("total_llm_calls", 0))
        kpi4.metric("Avg Memory Size", f"{aggregate.get('memory_size_tokens_avg', 0.0):.0f} tok")
        kpi5.metric("Total Cost", "$0.00", help="Open-source models run locally/Colab")

        st.subheader("Metrics vs. Paper Expected")
        obtained = aggregate.get("overall_metrics", {})
        exp_metrics = expected.get("metrics", {}) if isinstance(expected, dict) else {}

        rows = []
        for m_name, val in sorted(obtained.items()):
            exp_val = exp_metrics.get(m_name, "TODO")
            status = "TODO/BASELINE"
            diff = "-"
            if isinstance(exp_val, (int, float)):
                d = val - exp_val
                diff = f"{d:+.2f}"
                status = "✅ PASS" if d >= 0 else "❌ DIFF"
            rows.append({
                "Metric": m_name,
                "Obtained": val,
                "Paper Expected": str(exp_val),
                "Delta": diff,
                "Status": status,
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No aggregate metrics available yet for this run.")

        # Per Category Breakdown
        per_cat = aggregate.get("per_category_metrics", {})
        if per_cat:
            st.subheader("Category Breakdown")
            cat_rows = []
            for cat_name, metrics in sorted(per_cat.items()):
                entry = {"Category": cat_name, "Count": metrics.get("count", 0)}
                for k, v in metrics.items():
                    if k != "count":
                        entry[k] = v
                cat_rows.append(entry)
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # --- TAB 2: Item Inspector ---
    with tab_items:
        if not items:
            st.info("No individual evaluated items found in `items.jsonl`.")
        else:
            col_search, col_cat = st.columns([3, 1])
            search_query = col_search.text_input("Search Question or Item ID", "")
            categories = ["All"] + sorted(list({it.get("category", "") for it in items if it.get("category")}))
            selected_cat = col_cat.selectbox("Filter Category", categories)

            filtered_items = items
            if selected_cat != "All":
                filtered_items = [it for it in filtered_items if it.get("category") == selected_cat]
            if search_query:
                q_low = search_query.lower()
                filtered_items = [
                    it for it in filtered_items
                    if q_low in it.get("item_id", "").lower()
                    or q_low in it.get("prediction", "").lower()
                    or q_low in it.get("gold", "").lower()
                    or q_low in str(it.get("inputs", {})).lower()
                ]

            st.write(f"Showing **{len(filtered_items)}** of **{len(items)}** items:")

            for it in filtered_items[:25]:
                with st.expander(f"🔹 Item: `{it.get('item_id')}` | Category: `{it.get('category')}`"):
                    col_q, col_p, col_g = st.columns(3)
                    with col_q:
                        st.markdown("**Question / Prompt:**")
                        inputs = it.get("inputs", {})
                        prompt_text = inputs.get("question") or inputs.get("prompt") or str(inputs)
                        st.info(prompt_text)
                    with col_p:
                        st.markdown("**Prediction:**")
                        st.success(it.get("prediction", ""))
                    with col_g:
                        st.markdown("**Ground Truth Gold:**")
                        st.code(it.get("gold", ""))

                    # Per-item metrics
                    m_cols = st.columns(len(it.get("metrics", {})) + 2)
                    for idx, (m_k, m_v) in enumerate(it.get("metrics", {}).items()):
                        m_cols[idx].metric(m_k, f"{m_v:.1f}")
                    m_cols[-2].metric("Latency", f"{it.get('latency_seconds', 0.0):.2f}s")
                    m_cols[-1].metric("Tokens", it.get("tokens_used", 0))

    # --- TAB 3: Retrieval Trace Viewer ---
    with tab_trace:
        st.subheader("Retrieval Trace Visualizer")
        st.caption("Inspect the bidirectional chain traversal and self-reflection step-by-step.")

        if not items:
            st.info("No items available to inspect traces.")
        else:
            item_ids = [it.get("item_id") for it in items]
            inspect_item_id = st.selectbox("Select Item ID for Trace", item_ids)

            trace = load_item_trace(run_dir, inspect_item_id)
            if not trace:
                st.info(f"No detailed trace file found for `{inspect_item_id}` in `traces/` directory.")
            else:
                col_t1, col_t2 = st.columns(2)
                col_t1.markdown(f"**Query Categories Extracted:** `{trace.get('query_categories', [])}`")
                col_t2.markdown(f"**Retrieval Triggered:** `{trace.get('retrieval_triggered', True)}`")

                steps = trace.get("traversal_steps", [])
                st.markdown(f"**Traversal Steps:** `{len(steps)}`")

                for s_idx, step in enumerate(steps):
                    action = step.get("action", "unknown")
                    color = "green" if action == "adopt" else "orange"
                    st.markdown(
                        f"**Step {s_idx + 1}:** Node `{step.get('node_id')}` | "
                        f"Direction: `{step.get('direction')}` | "
                        f"Action: :{color}[**{action.upper()}**]"
                    )
                    st.text(f"Node Summary: {step.get('text', '')}")
                    st.divider()

    # --- TAB 4: Raw JSON Artifacts ---
    with tab_raw:
        st.subheader("Manifest JSON")
        st.json(manifest)
        if aggregate:
            st.subheader("Aggregate JSON")
            st.json(aggregate)


if __name__ == "__main__":
    main()
