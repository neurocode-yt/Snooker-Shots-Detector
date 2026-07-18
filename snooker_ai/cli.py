"""Command-line interface for snooker-ai."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from snooker_ai import __version__
from snooker_ai.config import load_config
from snooker_ai.evaluation.metrics import evaluate_dataset
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter
from snooker_ai.types import EditMode, ExportRequest, JobStatus, ShotUpdate
from snooker_ai.utils.logging import setup_logging

app = typer.Typer(
    name="snooker-ai",
    help="Automatic snooker shot detection and video editing.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _cfg(config: Optional[Path]):
    return load_config(config) if config else load_config()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"snooker-ai {__version__} (phase 1 baseline)")


@app.command()
def analyze(
    input_video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file"),
    mode: str = typer.Option(
        "strict",
        "--mode",
        "-m",
        help="strict | action_only | natural | full_sequence",
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config path"),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Reuse / set job id"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore previous analysis"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Analyze a snooker video and detect shots."""
    cfg = _cfg(config)
    cfg.ensure_dirs()
    store = JobStore(cfg)
    edit_mode = EditMode.from_string(mode)

    if job_id:
        jid = job_id
        job_dir = store.job_dir(jid)
        job_dir.mkdir(parents=True, exist_ok=True)
        if not (job_dir / "job.json").exists():
            meta = {
                "job_id": jid,
                "source_path": str(input_video.resolve()),
                "mode": edit_mode.value,
                "status": JobStatus.PENDING.value,
                "progress": 0.0,
            }
            (job_dir / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        jid = store.create(input_video, mode=edit_mode.value)

    job_dir = output_dir or store.job_dir(jid)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Job[/bold] {jid}")
    console.print(f"[bold]Mode[/bold] {edit_mode.value}")
    console.print(f"[bold]Input[/bold] {input_video}")

    analyzer = Analyzer(cfg, job_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=100)

        def on_prog(p: float, stage: str, msg: str) -> None:
            progress.update(task, completed=int(p * 100), description=f"{stage}: {msg}"[:60])
            store.update_progress(jid, p, stage, msg)

        try:
            result = analyzer.analyze(
                input_video,
                jid,
                mode=edit_mode,
                progress=on_prog,
                resume=not no_resume,
            )
            store.update_progress(
                jid,
                1.0,
                JobStatus.READY_FOR_REVIEW,
                f"Detected {len(result.shots)} shots",
                shots_detected=len(result.shots),
            )
        except Exception as exc:
            store.update_progress(jid, 0.0, JobStatus.FAILED, str(exc), error=str(exc))
            console.print(f"[red]Analysis failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    table = Table(title=f"Detected shots ({len(result.shots)})")
    table.add_column("ID", justify="right")
    table.add_column("Strike")
    table.add_column("Clip")
    table.add_column("Conf")
    table.add_column("Review")
    table.add_column("Replay")
    for s in result.shots:
        table.add_row(
            str(s.shot_id),
            f"{s.cue_strike:.2f}s",
            f"{s.clip_start:.2f}–{s.clip_end:.2f}",
            f"{s.shot_confidence:.2f}",
            "yes" if s.manual_review_required else "",
            "yes" if s.possible_replay else "",
        )
    console.print(table)
    console.print(
        f"Original {result.original_duration:.1f}s → edited ~{result.edited_duration:.1f}s "
        f"(removed ~{result.pause_removed_seconds:.1f}s)"
    )
    console.print(f"Results: {job_dir / 'analysis.json'}")
    console.print(f"Review:  snooker-ai review {jid}")
    console.print(f"Export:  snooker-ai export {jid} --output highlights.mp4")


@app.command()
def review(
    job_id: str = typer.Argument(..., help="Job id"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    web: bool = typer.Option(True, "--web/--no-web", help="Open web review UI"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """List shots for a job, or launch the web review interface."""
    cfg = _cfg(config)
    store = JobStore(cfg)
    try:
        result = store.load_analysis(job_id)
    except FileNotFoundError:
        console.print(f"[red]No analysis for job {job_id}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Job {job_id}")
    table.add_column("ID")
    table.add_column("Strike")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Conf")
    table.add_column("Included")
    for s in result.shots:
        table.add_row(
            str(s.shot_id),
            f"{s.cue_strike:.3f}",
            f"{s.clip_start:.3f}",
            f"{s.clip_end:.3f}",
            f"{s.shot_confidence:.2f}",
            str(s.included),
        )
    console.print(table)

    if web:
        console.print(f"Starting review server at http://{host}:{port}/review/{job_id}")
        import uvicorn
        from apps.api.main import create_app

        application = create_app(cfg)
        uvicorn.run(application, host=host, port=port, log_level="info")


@app.command()
def export(
    job_id: str = typer.Argument(...),
    output: Path = typer.Option(Path("highlights.mp4"), "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    fast: bool = typer.Option(False, "--fast", help="Stream-copy cuts when possible"),
    clips: bool = typer.Option(True, "--clips/--no-clips"),
    include_replays: bool = typer.Option(False, "--include-replays"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
) -> None:
    """Export individual clips and a joined highlights video."""
    cfg = _cfg(config)
    store = JobStore(cfg)
    result = store.load_analysis(job_id)
    if mode:
        from snooker_ai.pipeline.analyzer import Analyzer

        analyzer = Analyzer(cfg, store.job_dir(job_id))
        result = analyzer.resegment(result, EditMode.from_string(mode))

    out_dir = store.job_dir(job_id) / "export"
    request = ExportRequest(
        mode=result.mode,
        output_path=str(output.name if output.suffix else "highlights.mp4"),
        export_clips=clips,
        export_joined=True,
        include_replays=include_replays,
        accurate=not fast,
        min_confidence=min_confidence,
    )
    store.update_progress(job_id, 0.1, JobStatus.EXPORTING, "Exporting")
    exporter = Exporter(cfg)
    try:
        er = exporter.export(result, out_dir, request)
        # Copy joined to requested path if different
        if er.joined_path and output:
            dest = output if output.is_absolute() or output.parent != Path(".") else Path.cwd() / output
            if er.joined_path.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(er.joined_path, dest)
                console.print(f"[green]Wrote[/green] {dest}")
            else:
                console.print(f"[green]Wrote[/green] {er.joined_path}")
        console.print(f"Clips: {len(er.clip_paths)} in {out_dir / 'clips'}")
        store.update_progress(job_id, 1.0, JobStatus.COMPLETED, "Export complete")
    except Exception as exc:
        store.update_progress(job_id, 0.0, JobStatus.FAILED, str(exc), error=str(exc))
        console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def batch(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    mode: str = typer.Option("strict", "--mode", "-m"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    pattern: str = typer.Option("*.mp4", "--pattern"),
    export_after: bool = typer.Option(True, "--export/--no-export"),
) -> None:
    """Batch-analyze all matching videos in a directory."""
    cfg = _cfg(config)
    cfg.ensure_dirs()
    store = JobStore(cfg)
    edit_mode = EditMode.from_string(mode)

    videos = sorted(directory.rglob(pattern))
    if pattern == "*.mp4":
        for ext in ("*.mov", "*.mkv", "*.avi", "*.webm"):
            videos.extend(directory.rglob(ext))
        videos = sorted(set(videos))

    if not videos:
        console.print("[yellow]No videos found[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found {len(videos)} videos")
    for vid in videos:
        console.rule(str(vid.name))
        try:
            jid = store.create(vid, mode=edit_mode.value)
            analyzer = Analyzer(cfg, store.job_dir(jid))

            def on_prog(p: float, stage: str, msg: str, _jid: str = jid) -> None:
                store.update_progress(_jid, p, stage, msg)

            result = analyzer.analyze(vid, jid, mode=edit_mode, progress=on_prog, resume=True)
            store.update_progress(
                jid,
                1.0,
                JobStatus.READY_FOR_REVIEW,
                f"Detected {len(result.shots)} shots",
                shots_detected=len(result.shots),
            )
            console.print(f"[green]OK[/green] {vid.name} → {len(result.shots)} shots ({jid})")
            if export_after:
                out_dir = store.job_dir(jid) / "export"
                Exporter(cfg).export(
                    result,
                    out_dir,
                    ExportRequest(
                        mode=edit_mode,
                        output_path=f"{vid.stem}_highlights.mp4",
                        accurate=True,
                    ),
                )
                console.print(f"  exported → {out_dir}")
        except Exception as exc:
            console.print(f"[red]Failed {vid}:[/red] {exc}")


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("benchmark_report.json"), "--output", "-o"),
    tolerance: float = typer.Option(0.75, "--tolerance", help="Strike match tolerance (s)"),
) -> None:
    """Evaluate predictions against a ground-truth dataset."""
    report = evaluate_dataset(dataset, strike_tol=tolerance)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    console.print(json.dumps(report.to_dict(), indent=2))
    console.print(f"Wrote {output}")


@app.command()
def train(
    dataset_config: Path = typer.Argument(..., exists=True, help="Dataset YAML config"),
) -> None:
    """Launch training (Phase 2/3 scaffolding)."""
    console.print(
        "[yellow]Phase 1 ships a rule-based detector. "
        "Training entrypoint is scaffolded for Phase 2/3 models.[/yellow]"
    )
    console.print(f"Config: {dataset_config}")
    # Run scaffold script
    script = Path(__file__).resolve().parent.parent / "training" / "train.py"
    if script.exists():
        import runpy

        sys.argv = ["train.py", str(dataset_config)]
        runpy.run_path(str(script), run_name="__main__")
    else:
        console.print("[red]training/train.py not found[/red]")
        raise typer.Exit(1)


@app.command("update-shot")
def update_shot_cmd(
    job_id: str,
    shot_id: int,
    clip_start: Optional[float] = typer.Option(None),
    clip_end: Optional[float] = typer.Option(None),
    cue_strike: Optional[float] = typer.Option(None),
    included: Optional[bool] = typer.Option(None),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Manually update a shot boundary from the CLI."""
    cfg = _cfg(config)
    store = JobStore(cfg)
    s = store.update_shot(
        job_id,
        shot_id,
        ShotUpdate(
            clip_start=clip_start,
            clip_end=clip_end,
            cue_strike=cue_strike,
            included=included,
        ),
    )
    console.print(s.model_dump_json(indent=2))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Start the REST API and web review interface."""
    cfg = _cfg(config)
    import uvicorn
    from apps.api.main import create_app

    application = create_app(cfg)
    console.print(f"API + review UI at http://{host}:{port}")
    uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
