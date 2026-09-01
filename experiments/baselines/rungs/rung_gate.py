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

root = Path(sys.argv[1])
want_rows = int(sys.argv[2])
want_gpus = int(sys.argv[3])
want_hosts = int(sys.argv[4])
want_system = sys.argv[5] if len(sys.argv) > 5 else None
fails = []


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

energies = {}
for d in sorted(glob.glob(str(root / "results" / "*"))):
    csv = Path(d) / "run" / "train_stats.csv"
    if csv.exists():
        energies[os.path.basename(d)] = csv.read_text().strip().splitlines()[-1].split(",")[1]
print(f"GATE energies         {energies}")
if len(energies) != want_rows:
    fails.append(f"{len(energies)} of {want_rows} rows produced train_stats.csv")
elif len(set(energies.values())) != want_rows:
    fails.append(f"NON-DISTINCT energies -> seeds not consumed: {energies}")

# WRONG-SYSTEM CHECK. Structural gates cannot see this.
if want_system:
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

if fails:
    print("RUNG_GATES_FAILED")
    for f in fails:
        print(f"  FAIL: {f}")
    sys.exit(1)
print("RUNG_GATES_PASSED")
