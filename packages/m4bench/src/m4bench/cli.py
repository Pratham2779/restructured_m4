"""Command-line interface for m4bench."""
import os
import sys
import json
import importlib
from typing import Optional, List
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
def main():
    """m4bench: Evaluation & Benchmark Suite for Meta-Memory (M4)."""
    pass


@main.command(name="run")
@click.argument("dataset")
@click.option("--stage", type=click.Choice(["smoke", "pilot", "full"]), default="smoke", help="Execution stage: smoke (1-2 items), pilot (subsample), or full.")
@click.option("--workers", type=int, default=1, help="Number of worker threads (default 1).")
@click.option("--gamma", type=float, default=0.5, help="Cosine similarity threshold gamma (default 0.5).")
@click.option("--temperature", type=float, default=0.8, help="Generation temperature (default 0.8).")
@click.option("--model", type=str, default="qwen2.5:7b-instruct", help="LLM model identifier.")
@click.option("--embedder", type=str, default="bge-small-en-v1.5", help="Embedding model identifier.")
@click.option("--seed", type=int, default=42, help="Random seed for sampling.")
@click.option("--results-dir", type=str, default="results", help="Directory where results will be stored.")
@click.option("--ablate", type=str, multiple=True, help="Components to ablate (e.g. chain, calibrate, compress, self_reflection).")
@click.option("--mock", is_flag=True, default=False, help="Run with deterministic fake models (offline hermetic execution).")
def run_command(
    dataset: str,
    stage: str,
    workers: int,
    gamma: float,
    temperature: float,
    model: str,
    embedder: str,
    seed: int,
    results_dir: str,
    ablate: tuple,
    mock: bool,
):
    """Run an evaluation protocol for a specific dataset."""
    console.print(f"[bold cyan]Starting m4bench evaluation for dataset:[/bold cyan] {dataset} (stage: {stage})")
    
    # Ensure current working directory is in sys.path so benchmarks can be imported
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module_name = f"benchmarks.{dataset.lower()}.protocol"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Could not load protocol module '{module_name}': {e}")
        console.print(f"Make sure you are running from repo root and 'benchmarks/{dataset.lower()}/protocol.py' exists.")
        sys.exit(1)

    if not hasattr(mod, "run_benchmark"):
        console.print(f"[bold red]Error:[/bold red] Protocol module '{module_name}' must expose a 'run_benchmark' function.")
        sys.exit(1)

    try:
        agg = mod.run_benchmark(
            stage=stage,
            workers=workers,
            gamma=gamma,
            temperature=temperature,
            model=model,
            embedder=embedder,
            seed=seed,
            results_dir=results_dir,
            ablations=list(ablate),
            mock=mock,
        )
        console.print(f"[bold green]Run completed successfully.[/bold green] Total items evaluated: {agg.total_items}")
    except Exception as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise exc


@main.command(name="report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--expected-file", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None, help="Explicit path to expected.yaml file.")
def report_command(run_dir: str, expected_file: Optional[str]):
    """Generate a terminal summary comparing run results against expected paper numbers."""
    manifest_path = os.path.join(run_dir, "manifest.json")
    aggregate_path = os.path.join(run_dir, "aggregate.json")

    if not os.path.exists(aggregate_path):
        console.print(f"[bold red]Error:[/bold red] '{aggregate_path}' not found in run directory.")
        sys.exit(1)

    with open(aggregate_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Locate expected.yaml if not provided
    expected = {}
    if expected_file:
        with open(expected_file, "r", encoding="utf-8") as f:
            expected = yaml.safe_load(f) or {}
    else:
        dataset = agg.get("dataset", "")
        candidates = [
            os.path.join(run_dir, "expected.yaml"),
            os.path.join("benchmarks", dataset, "expected.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                with open(c, "r", encoding="utf-8") as f:
                    expected = yaml.safe_load(f) or {}
                break

    # Header Panel
    header_text = (
        f"[bold]Dataset:[/bold] {agg.get('dataset', 'Unknown')}  |  "
        f"[bold]Run ID:[/bold] {agg.get('run_id')}  |  "
        f"[bold]Stage:[/bold] {agg.get('stage')}\n"
        f"[bold]Model:[/bold] {manifest.get('model_name', 'N/A')}  |  "
        f"[bold]Embedder:[/bold] {manifest.get('embedder_name', 'N/A')}  |  "
        f"[bold]Gamma (γ):[/bold] {manifest.get('gamma', 'N/A')}  |  "
        f"[bold]Total Items:[/bold] {agg.get('total_items', 0)}"
    )
    console.print(Panel(header_text, title="[bold cyan]M4 Benchmark Run Report[/bold cyan]", border_style="cyan"))

    # Overall Metrics Table
    overall_table = Table(title="Overall Metrics Comparison", show_header=True, header_style="bold magenta")
    overall_table.add_column("Metric", style="dim", width=20)
    overall_table.add_column("Obtained", justify="right")
    overall_table.add_column("Paper Expected", justify="right")
    overall_table.add_column("Status", justify="center")

    obtained_metrics = agg.get("overall_metrics", {})
    expected_metrics = expected.get("metrics", {}) if isinstance(expected, dict) else {}

    for metric_name, val in sorted(obtained_metrics.items()):
        exp_val = expected_metrics.get(metric_name, "TODO")
        status = "[yellow]BASELINE/TODO[/yellow]"
        if isinstance(exp_val, (int, float)):
            if val >= exp_val:
                status = "[green]✓ PASS[/green]"
            else:
                status = "[red]✗ DIFF[/red]"
        overall_table.add_row(metric_name, str(val), str(exp_val), status)

    console.print(overall_table)

    # Category Breakdown Table (if available)
    per_cat = agg.get("per_category_metrics", {})
    if per_cat:
        cat_table = Table(title="Per-Category Metrics", show_header=True, header_style="bold blue")
        cat_table.add_column("Category", style="bold", width=25)
        cat_table.add_column("Count", justify="right")
        # Find unique metric names across categories
        all_cat_metrics = set()
        for cat_data in per_cat.values():
            all_cat_metrics.update(k for k in cat_data.keys() if k != "count")
        for m in sorted(all_cat_metrics):
            cat_table.add_column(m, justify="right")

        for cat_name, cat_data in sorted(per_cat.items()):
            row = [cat_name, str(cat_data.get("count", 0))]
            for m in sorted(all_cat_metrics):
                row.append(str(cat_data.get(m, "-")))
            cat_table.add_row(*row)

        console.print(cat_table)

    # Telemetry Panel
    telemetry_text = (
        f"Avg Latency: {agg.get('avg_latency_seconds', 0.0):.2f}s  |  "
        f"Avg Tokens/Item: {agg.get('avg_tokens_per_item', 0.0):.1f}\n"
        f"Total Prompt Tokens: {agg.get('total_prompt_tokens', 0):,}  |  "
        f"Total Completion Tokens: {agg.get('total_completion_tokens', 0):,}\n"
        f"Avg Memory Size (Tokens): {agg.get('memory_size_tokens_avg', 0.0):.1f}"
    )
    console.print(Panel(telemetry_text, title="Telemetry & Resource Usage", border_style="dim"))

    # Known Deviations
    deviations = agg.get("known_deviations", [])
    if deviations:
        dev_text = "\n".join(f"• {d}" for d in deviations)
        console.print(Panel(dev_text, title="[yellow]Known Protocol Deviations[/yellow]", border_style="yellow"))


@main.command(name="sweep-gamma")
@click.option("--dataset", type=str, default="locomo", help="Dataset name (default 'locomo').")
@click.option("--stage", type=click.Choice(["smoke", "pilot"]), default="pilot", help="Stage for sweep (default 'pilot').")
@click.option("--gammas", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8", help="Comma-separated gamma values.")
@click.option("--results-dir", type=str, default="results/sweeps", help="Output directory for sweep runs.")
def sweep_gamma_command(dataset: str, stage: str, gammas: str, results_dir: str):
    """Reproduce Figure 9: Evaluate sensitivity over gamma thresholds."""
    gamma_list = [float(g.strip()) for g in gammas.split(",") if g.strip()]
    console.print(f"[bold cyan]Starting Gamma Sweep on '{dataset}' across:[/bold cyan] {gamma_list}")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module_name = f"benchmarks.{dataset.lower()}.protocol"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Could not load protocol module '{module_name}': {e}")
        sys.exit(1)

    sweep_results = {}
    for g in gamma_list:
        console.print(f"\n[bold yellow]>>> Running gamma = {g}...[/bold yellow]")
        run_res_dir = os.path.join(results_dir, f"gamma_{g}")
        agg = mod.run_benchmark(
            stage=stage,
            gamma=g,
            results_dir=run_res_dir,
        )
        sweep_results[g] = agg.overall_metrics

    # Display sweep comparison table
    sweep_table = Table(title=f"Figure 9 Gamma Sensitivity Sweep ({dataset.upper()} - {stage})", header_style="bold green")
    sweep_table.add_column("Gamma (γ)", justify="center", style="bold")
    
    all_metrics = set()
    for m in sweep_results.values():
        all_metrics.update(m.keys())
    for m in sorted(all_metrics):
        sweep_table.add_column(m, justify="right")

    for g in gamma_list:
        row = [f"{g:.2f}"]
        for m in sorted(all_metrics):
            row.append(str(sweep_results[g].get(m, "-")))
        sweep_table.add_row(*row)

    console.print(sweep_table)


if __name__ == "__main__":
    main()
