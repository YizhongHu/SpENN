"""Shared artefact gate for the Polaris toolkit-dispatch ladder (L0..L3).

Usage: rung_gate.py <run_root> <expected_rows> <expected_gpus_per_row> <expected_hosts>

WHY THIS EXISTS AS A FILE: L0 failed its gate three times for reasons that were
all in the GATE, not the run. Each schema fact below cost a queue cycle to learn,
so they are written down once rather than re-derived per rung.

SCHEMA FACTS, none of which are guessable:
  * dispatch_records.jsonl entries are POINTERS, not outcomes. They carry
    `status_path` / `metadata.attempt_status_path`; the row's real status lives
    in that file. Reading record["status"] silently yields None.
  * The GPU binding evidence is `inherited_visibility_value`, NOT
    `visibility_value`. The latter is None BY DESIGN: parsl_attach.py:265 --
    Parsl's `available_accelerators` binds each HTEX worker and the task inherits
    that environment, because dispatch order must never select a GPU.
  * `placement.gpus` lists every GPU on the NODE (nvidia-smi ignores
    CUDA_VISIBLE_DEVICES), so its length is 4 regardless of binding. It cannot
    prove per-row exclusivity; `inherited_visibility_value` can.
  * verification.json exit_code is NOT a verdict: pipeline.py writes 0 whenever
    dispatch() does not raise, and a failed row is recorded, not raised.
"""
import json, sys, glob, os
from pathlib import Path

# Expected energy band per system, in hartree. A wrong-SYSTEM run passes every
# STRUCTURAL gate identically -- records, hosts, visibility, checkpoints, even
# distinct energies -- because it is a perfectly good run of the wrong thing.
# Only the energy separates them.
# This exists because production job 7579539 ran H2 while labelled He: the plan
# builder was substituted on the cluster copy but not the local one used to
# derive the production script. It was caught by an UNRELATED node fault, not by
# this check, which at the time lived only in a watch prohibition and in prose.
ENERGY_BAND = {
    "he": (-2.95, -2.85),
    "h2": (-1.22, -1.13),
    "li": (-7.52, -7.43),
    "be": (-14.72, -14.61),
    "b":  (-24.71, -24.60),
    "n":  (-54.65, -54.53),
}

#: The energy band can only judge a CONVERGED run. A 200- or 3000-step ramp rung
#: sits far above its band and would fail for the right reason at the wrong time.
#: So the band is gated on step count -- and the rungs below that threshold are
#: exactly the ones that most need a system check, which is why the command-based
#: check below exists and does not depend on convergence at all.
#: How many leading rows form a run's fingerprint for the seeds-consumed test.
#: A row whose seed was ignored reproduces another row's trajectory EXACTLY, so it
#: collides on all TRAJECTORY_PREFIX values at once. Independent rows colliding on
#: that many values by chance is not a practical concern.
TRAJECTORY_PREFIX = 10

BAND_MIN_STEPS = 50000

#: Which flag names the system, per config style. Atoms carry an element symbol;
#: H2 carries a molecule name. Reading either is enough to identify the system
#: from argv alone.
SYSTEM_FLAGS = ("--config.system.atom", "--config.system.molecule_name")

def _flag(argv, name):
    """Value following `name` in argv, or None. argv may be a list or a string."""
    if isinstance(argv, str):
        argv = argv.split()
    try:
        return argv[list(argv).index(name) + 1]
    except (ValueError, IndexError):
        return None


root = Path(sys.argv[1])
want_rows = int(sys.argv[2])
want_gpus = int(sys.argv[3])
want_hosts = int(sys.argv[4])
want_system = sys.argv[5] if len(sys.argv) > 5 else None
want_ansatz = sys.argv[6] if len(sys.argv) > 6 else None
fails = []
ran, skipped = [], []



recs_path = root / "launch" / "dispatch_records.jsonl"
recs = []
if recs_path.exists():
    recs = [json.loads(l) for l in recs_path.read_text().splitlines() if l.strip()]
else:
    fails.append("dispatch_records.jsonl MISSING (a raised completion predicate suppresses it entirely)")

statuses, inherited, hosts, placements = [], [], set(), []
for r in recs:
    p = (r.get("metadata") or {}).get("attempt_status_path") or r.get("status_path")
    try:
        s = json.load(open(p))
    except Exception as exc:
        fails.append(f"attempt_status unreadable for {r.get('run_id')}: {exc!r}")
        continue
    statuses.append(s.get("status"))
    vis = s.get("inherited_visibility_value")
    host = (s.get("placement") or {}).get("hostname")
    inherited.append(vis)
    hosts.add(host)
    placements.append((host, vis))

print(f"GATE records          {len(recs)} / {want_rows}")
print(f"GATE statuses         {statuses}")
print(f"GATE inherited_vis    {inherited}")
print(f"GATE host_gpu_pairs    {len(set(placements))} distinct of {len(placements)}")
print(f"GATE hosts            {sorted(h for h in hosts if h)}")

if len(recs) != want_rows:
    fails.append(f"record count {len(recs)} != {want_rows}")
bad = [i for i, s in enumerate(statuses) if s != "success"]
if bad:
    fails.append(f"rows not success at indices {bad}: {[statuses[i] for i in bad]}")

present = [v for v in inherited if v is not None]
# Exclusivity is per (HOST, visibility), NOT per visibility. Across N nodes the
# same local id recurs once per node -- GPU "0" on node A is a different physical
# device from GPU "0" on node B. Requiring globally distinct values is an
# invariant that holds only at one node, and it produced a false DOUBLE-BOOKED
# verdict on L2b (8 rows, 2 nodes, values ['1','0','2','3','2','0','3','1'] --
# every one correct).
pairs = [pl for pl in placements if pl[1] is not None and pl[0] is not None]
if len(present) != len(recs):
    fails.append(f"inherited visibility missing for {len(recs)-len(present)} row(s)")
elif len(set(pairs)) != len(pairs):
    dupes = sorted({pl for pl in pairs if pairs.count(pl) > 1})
    fails.append(f"DOUBLE-BOOKED: (host, visibility) pairs repeat: {dupes}")
else:
    widths = {len(v.split(",")) for v in present}
    if widths != {want_gpus}:
        fails.append(f"gpus-per-row {sorted(widths)} != {want_gpus}")

if len([h for h in hosts if h]) != want_hosts:
    fails.append(f"host count {len([h for h in hosts if h])} != {want_hosts}")

energies, fingerprints = {}, {}
for d in sorted(glob.glob(str(root / "results" / "*"))):
    csv = Path(d) / "run" / "train_stats.csv"
    if csv.exists():
        name = os.path.basename(d)
        lines = csv.read_text().strip().splitlines()
        energies[name] = lines[-1].split(",")[1]
        # Skip the header, take the earliest steps: they are set by the seed's
        # initial walkers and have not yet been pulled together by optimisation.
        fingerprints[name] = tuple(l.split(",")[1] for l in lines[1:1 + TRAJECTORY_PREFIX])
print(f"GATE energies         {energies}")
if len(energies) != want_rows:
    fails.append(f"{len(energies)} of {want_rows} rows produced train_stats.csv")
else:
    # SEEDS-CONSUMED TEST, on the trajectory rather than on the final energy.
    #
    # Testing final energies for distinctness looks equivalent and is not. Final
    # energies CONVERGE: as a run improves, all rows crowd toward the same value,
    # so the probability that two of them round to the same printed number RISES
    # with run quality. That check is therefore most likely to fire spuriously on
    # the best runs -- an alarm whose rate is governed by convergence, not by
    # correctness. R2 arm lo (job 7580172) tripped it at 2000 steps: seed-11 and
    # seed-14 both printed -2.9036388, while differing from step 0 onward and
    # agreeing on exactly 1 of 2000 rows. The n=20 production run 7579717 passed
    # the same check by luck.
    dupes = {}
    for name, fp in fingerprints.items():
        dupes.setdefault(fp, []).append(name)
    collided = [names for names in dupes.values() if len(names) > 1]
    if collided:
        fails.append(
            f"SEEDS NOT CONSUMED: rows share their first {TRAJECTORY_PREFIX} "
            f"energies, i.e. identical trajectories: {collided}")
    # Report final-energy coincidences, but do NOT fail on them: see above.
    if len(set(energies.values())) != want_rows:
        same = {}
        for n_, v in energies.items():
            same.setdefault(v, []).append(n_)
        print("GATE note             final-energy coincidence (NOT a failure; "
              f"trajectories verified distinct): "
              f"{[g for g in same.values() if len(g) > 1]}")

# COMMAND-BASED SYSTEM/ANSATZ CHECK.
#
# This reads what was SUBMITTED, and it is valid at any step count -- 200 steps
# or 200000. It exists because the energy band, the only previous system check,
# needs a converged run, so the short rungs had no system check at all.
#
# Preference order matters. `submitted_command` is what dispatch actually sent;
# `plan/tasks.jsonl` is only what was intended. They can differ. But one failed
# row makes dispatch raise and suppresses ALL records, so records may be absent
# on a run whose rows were fine -- hence the fallback, which announces itself
# rather than quietly weakening the check.
cmd_source, cmds = None, []
if recs:
    cmds = [r.get("submitted_command") or [] for r in recs]
    cmd_source = "dispatch_records.submitted_command"
if not any(cmds):
    plan_tasks = root / "plan" / "tasks.jsonl"
    if plan_tasks.exists():
        cmds = [json.loads(l).get("command") or []
                for l in plan_tasks.read_text().splitlines() if l.strip()]
        cmd_source = "plan/tasks.jsonl (FALLBACK: dispatch records absent or empty)"

observed_steps = set()
if not any(cmds):
    # Never silently skip: with no command anywhere, this check cannot run, and
    # saying nothing would read as a pass.
    fails.append("no submitted_command and no plan/tasks.jsonl -- command-based "
                 "system/ansatz check COULD NOT RUN")
else:
    print(f"GATE command source   {cmd_source} ({len([c for c in cmds if c])} row(s))")
    got_sys, got_ans = set(), set()
    for argv in cmds:
        if not argv:
            continue
        for flag in SYSTEM_FLAGS:
            v = _flag(argv, flag)
            if v is not None:
                got_sys.add(v.lower())
        v = _flag(argv, "--config.network.network_type")
        if v is not None:
            got_ans.add(v.lower())
        v = _flag(argv, "--config.optim.iterations")
        if v is not None:
            observed_steps.add(int(v))
    print(f"GATE argv system      {sorted(got_sys)}")
    print(f"GATE argv ansatz      {sorted(got_ans)}")
    print(f"GATE argv steps       {sorted(observed_steps)}")
    if len(got_sys) > 1:
        fails.append(f"MIXED SYSTEMS in one rung: {sorted(got_sys)}")
    if len(got_ans) > 1:
        fails.append(f"MIXED ANSATZES in one rung: {sorted(got_ans)}")
    if want_system:
        ran.append("argv-system")
        if got_sys and got_sys != {want_system.lower()}:
            fails.append(f"WRONG SYSTEM IN ARGV: asked for {want_system!r}, "
                         f"command says {sorted(got_sys)}")
        elif not got_sys:
            fails.append("command names no system flag; cannot confirm system from argv")
    if want_ansatz:
        ran.append("argv-ansatz")
        if got_ans and got_ans != {want_ansatz.lower()}:
            fails.append(f"WRONG ANSATZ IN ARGV: asked for {want_ansatz!r}, "
                         f"command says {sorted(got_ans)}")
        elif not got_ans:
            fails.append("command names no network_type; cannot confirm ansatz from argv")

# WRONG-SYSTEM CHECK BY ENERGY. Production only -- see BAND_MIN_STEPS.
band_applicable = bool(observed_steps) and min(observed_steps) >= BAND_MIN_STEPS
if want_system and not band_applicable:
    skipped.append(
        f"energy-band (steps {sorted(observed_steps) or 'unknown'} < {BAND_MIN_STEPS}; "
        "a short rung is not converged, so the band cannot judge it)")
if want_system and band_applicable:
    ran.append("energy-band")
if want_system and band_applicable:
    band = ENERGY_BAND.get(want_system.lower())
    if band is None:
        fails.append(f"no energy band known for system {want_system!r}; add one rather than skipping")
    else:
        lo, hi = band
        bad = {k: v for k, v in energies.items() if not (lo <= float(v) <= hi)}
        print(f"GATE system            {want_system} expects [{lo}, {hi}]")
        if bad:
            other = [s for s, (l, h) in ENERGY_BAND.items() if any(l <= float(v) <= h for v in bad.values())]
            fails.append(
                f"WRONG SYSTEM: {len(bad)} row(s) outside the {want_system} band {band}: "
                f"{dict(list(bad.items())[:3])}"
                + (f" -- these look like {other}" if other else "")
            )

# State what ran and what did not. A gate that skips a check silently is
# indistinguishable from one that passed it.
print(f"GATE checks ran       {ran or ['(none)']}")
if skipped:
    for sk in skipped:
        print(f"GATE checks SKIPPED   {sk}")

if fails:
    print("RUNG_GATES_FAILED")
    for f in fails:
        print(f"  FAIL: {f}")
    sys.exit(1)
print("RUNG_GATES_PASSED")
