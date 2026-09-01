"""Build a StagePlanV2 of independent ferminet/psiformer rows, one per seed.

SYSTEM IS AN ARGUMENT, NOT A COPY. An earlier design kept one builder per system
(`rung_makeplan.py` for H2, `rung_makeplan_he.py` for He) differing in four
lines. Production job 7579539 then ran H2 while labelled He, because the
production script was derived from a local copy that had never received the
He substitution -- applied remotely with `sed -i` -- and the pre-submission
check invoked the He builder BY NAME, validating a code path the job never took.
A single parameterised builder removes that class of error: there is nothing to
pick wrongly.

`build_plan` gives every row the SAME argv (test_plan.py:33 asserts it), which
would run four identical seeds -- exactly the defect that silently produced
identical rows on Cannon.  The dispatch layer does support per-row argv
(`admit_tasks(argv_by_runtime=...)`: "one exact argv per entry of tasks"), so
the plan is constructed directly with per-row LogicalTaskSpec commands.

L0 tests DISPATCH, not science: 300 steps is enough to prove four rows ran
independently on four distinct GPUs with four distinct seeds.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/rhu/src/TPEN-l0-dev")
from experiments.toolkit.dispatch import StagePlanV2, LogicalTaskSpec, CompletionSpec, logical_task_id_from_parts
from experiments.toolkit.resources import ResourceSpec


#: Per-system command flags. H2 pins its geometry explicitly: `diatomic.py`
#: defaults to 0.737164 angstrom and the reference rows were produced at
#: 1.4 bohr. Accepting the default is SILENT and worth ~2.1e-4 Ha, roughly six
#: times the seed standard deviation. Atoms take no geometry override.
SYSTEMS = {
    "h2": {
        "config_flags": lambda fn: (
            "--config", f"{fn}/ferminet/configs/diatomic.py",
            "--config.system.molecule_name", "H2",
            "--config.system.units", "bohr",
            "--config.system.bond_length", "1.4",
        ),
    },
    "he": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "He")},
    "li": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "Li")},
    "be": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "Be")},
    "b":  {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "B")},
    "n":  {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "N")},
}

FN = "/home/rhu/src/ferminet-psiformer-60c9fab3"
PY = "/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python"
SYSTEM = sys.argv[1].lower()
if SYSTEM not in SYSTEMS:
    raise SystemExit(f"unknown system {SYSTEM!r}; known: {sorted(SYSTEMS)}")
ROOT = sys.argv[2]
PLAN_DIR = sys.argv[3]
GPUS = int(sys.argv[4])
NROWS = int(sys.argv[5])
STEPS = int(sys.argv[6])
PLAN_ID = sys.argv[7]

tasks = []
for seed in range(NROWS):
    run_id = f"row-seed{seed}"
    result_dir = str(Path(ROOT) / f"seed-{seed}")
    # Completion must key on a file the RUNNER ACTUALLY WRITES. The first L0
    # attempt used policy="status_completed" against a status.json that ferminet
    # never emits: all four rows succeeded, the predicate failed, dispatch raised,
    # and dispatch_records was never written -- so a fully successful run reported
    # records=0. The predicate has to match the program, not the convention.
    done_path = str(Path(result_dir) / "run" / "train_stats.csv")
    cmd = (
        PY, f"{FN}/ferminet/main.py",
        *SYSTEMS[SYSTEM]["config_flags"](FN),
        "--config.batch_size", "4096",
        "--config.optim.iterations", str(STEPS),
        "--config.network.network_type", "psiformer",
        "--config.network.determinants", "16",
        "--config.debug.deterministic", "True",
        "--config.debug.seed", str(seed),
        "--config.log.save_path", f"{result_dir}/run",
    )
    tasks.append(LogicalTaskSpec(
        logical_task_id=logical_task_id_from_parts(stage="baselines", run_id=run_id, plan_id=PLAN_ID),
        stage="baselines",
        run_id=run_id,
        command=cmd,
        result_dir=result_dir,
        outputs=(done_path,),
        logs=(done_path,),
        params={"code": "ferminet", "ansatz": "psiformer", "system": SYSTEM,
                "seed": seed, "steps": STEPS, "batch_size": 4096},
        resources=ResourceSpec(profile="cuda", device="cuda", gpus=GPUS),
        completion=CompletionSpec(policy="file_exists", output_paths=(done_path,)),
        metadata={"runtime": "ferminet", "row_key": f"{SYSTEM}-seed-{seed}"},
    ))

plan = StagePlanV2(study="baselines", stage="baselines", plan_id=PLAN_ID,
                   results_root=ROOT, tasks=tuple(tasks)).validate()
Path(PLAN_DIR).mkdir(parents=True, exist_ok=True)
plan.write(PLAN_DIR)
# Assert the thing build_plan would have got wrong.
cmds = {t.command for t in plan.tasks}
assert len(cmds) == NROWS, f"EXPECTED {NROWS} DISTINCT COMMANDS, got {len(cmds)}"
seeds = sorted(t.params["seed"] for t in plan.tasks)
assert seeds == list(range(NROWS)), seeds
assert {t.resources.gpus for t in plan.tasks} == {GPUS}
print(f"PLAN_OK system={SYSTEM} tasks={len(plan.tasks)} distinct_commands={len(cmds)} gpus_per_row={GPUS} seeds={seeds}")
print(f"PLAN_DIR={PLAN_DIR}")
