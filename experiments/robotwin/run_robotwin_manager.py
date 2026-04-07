"""
Cosmos-WAM RoboTwin evaluation manager for multi-GPU parallel execution.

This manager handles running evaluation across all RoboTwin tasks in parallel,
similar to FastWAM's run_robotwin_manager.py.

Usage:
    python experiments/robotwin/run_robotwin_manager.py \
        ckpt=/path/to/checkpoint.pt \
        EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
        EVALUATION.text_embedding_cache_dir=/path/to/text_embeds \
        MULTIRUN.num_gpus=4

Or evaluate a single task:
    python experiments/robotwin/run_robotwin_manager.py \
        ckpt=/path/to/checkpoint.pt \
        EVALUATION.task_name=click_alarmcoin \
        EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
"""

import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

# Default task list if _eval_step_limit.yml not found
DEFAULT_ROBOTWIN_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
]


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    """Generate checkpoint tag from path."""
    parts = ckpt_path.resolve().parts
    if "outputs" in parts:
        outputs_idx = parts.index("outputs")
        if outputs_idx + 1 >= len(parts):
            return ckpt_path.stem
        return parts[outputs_idx + 1]
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    """Check if override should be blocked from worker."""
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    """Collect overrides to pass to worker processes."""
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks(robotwin_root: Path) -> list[str]:
    """Load task list from RoboTwin config or use default."""
    eval_step_limit_file = robotwin_root / "task_config" / "_eval_step_limit.yml"
    
    if eval_step_limit_file.exists():
        with eval_step_limit_file.open("r", encoding="utf-8") as f:
            task_map = yaml.safe_load(f)
        if isinstance(task_map, dict) and len(task_map) > 0:
            tasks = list(task_map.keys())
            # Remove duplicates while preserving order
            seen = set()
            dedup_tasks: list[str] = []
            for task in tasks:
                if task not in seen:
                    seen.add(task)
                    dedup_tasks.append(task)
            return dedup_tasks
    
    # Fallback to default tasks
    return DEFAULT_ROBOTWIN_TASKS


def _parse_success_rate(result_file: Path) -> float:
    """Parse success rate from result file."""
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _mean_or_none(values: list[float | None]) -> float | None:
    """Calculate mean, ignoring None values."""
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    """Convert to JSON-serializable value."""
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    """State of a running evaluation process."""
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")
    
    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    
    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    
    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))
    
    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"
    
    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks(robotwin_root)
    else:
        tasks = [str(task_name_cfg)]
    
    extra_overrides = _collect_worker_overrides()
    
    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []
    
    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }
    
    def log(msg: str) -> None:
        """Log message to stdout and file."""
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    
    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        """Build command for worker process."""
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd
    
    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        """Launch evaluation for a task phase."""
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(f"launch task={task_name} phase={phase} gpu={gpu_id} cmd={' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )
    
    def terminate_all_running() -> None:
        """Terminate all running processes."""
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()
    
    def gpu_running_count(gpu_id: int) -> int:
        """Count running tasks on a GPU."""
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count
    
    def try_launch_pending(gpu_id: int) -> None:
        """Try to launch pending tasks on a GPU."""
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))
    
    def write_outputs() -> None:
        """Write summary files."""
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])
        
        # CSV summary
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow([
                    task,
                    task_rates[task]["clean"],
                    task_rates[task]["random"],
                ])
            writer.writerow(["__overall__", clean_mean, random_mean])
        
        # JSON summary
        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # Failed tasks
        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )
    
    log(f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}")
    
    # Launch initial tasks
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)
    
    has_failure = False
    failure_message = ""
    
    # Main loop
    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)
            
            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append({
                    "task_name": state.task_name,
                    "phase": state.phase,
                    "gpu_id": gpu_id,
                    "return_code": return_code,
                    "reason": "process_failed",
                })
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break
            
            # Parse result
            result_file = run_output_dir / state.task_name / f"_result_{state.phase}.txt"
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append({
                    "task_name": state.task_name,
                    "phase": state.phase,
                    "gpu_id": gpu_id,
                    "return_code": return_code,
                    "reason": "result_parse_failed",
                })
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break
            
            task_rates[state.task_name][state.phase] = success_rate
            log(f"done task={state.task_name} phase={state.phase} gpu={gpu_id} success_rate={success_rate:.4f}")
            
            # If clean done, launch random
            if state.phase == "clean":
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase="random",
                ))
                continue
            
            # Launch next pending task
            try_launch_pending(gpu_id)
        
        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)
    
    # Mark not started tasks as failed
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append({
                "task_name": task_name,
                "phase": "not_started",
                "gpu_id": -1,
                "return_code": -1,
                "reason": "aborted_not_started",
            })
    
    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")
    
    if has_failure:
        raise RuntimeError(failure_message)
    
    log("manager finished successfully")


if __name__ == "__main__":
    main()
