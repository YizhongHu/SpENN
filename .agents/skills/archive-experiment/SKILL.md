---
name: archive-experiment
description: Archive one explicitly selected, completed TPEN experiment or run from Cannon Netscratch to ALCF HPSS with Globus. Use when asked to archive or back up an experiment/run to HPSS. This is an interactive, manual-first archive procedure; it never creates unattended schedules, retention rotation, deletion, or a broad Netscratch mirror.
---

# Archive experiment

Archive exactly one completed, immutable TPEN experiment or run through Globus:

```text
Cannon Netscratch → Harvard FAS RC Holyoke → Globus → alcf#dtn_hpss → ALCF HPSS
```

This skill is for a durable, append-only archive copy. It is not a snapshot system, a synchronizing mirror, or a cleanup tool.

## Non-negotiable safety rules

- Read `cluster-access` and the full current Task Orchestrator notes for Cannon and ALCF HPSS before touching either facility or Globus.
- Use the local Globus CLI. Do not install Globus or route a transfer through a Cannon login node.
- Require an exact source directory and durable completion evidence. Never archive `/n/netscratch/.../rhu` broadly, infer completion from a directory name, or recursively probe arbitrary trees.
- The source must be immutable before submission. A changed completed source is a lifecycle failure, not a reason to overwrite an archive.
- Do not submit a transfer until the user has approved the exact source and destination after reviewing a dry run.
- Never use `--delete-destination-extra`, `globus rm`, `globus mkdir`, HPSS deletion, retention pruning, or a personal token in a cron job.
- Do not claim a quota or retention guarantee. Record facility policy and actual task evidence instead.

## Preconditions

1. Confirm the user has completed collection-specific Globus consent and that `globus whoami` succeeds.
2. Revalidate the current production collection names and IDs from the facility notes. The route validated on 2026-08-20 was:

   ```text
   source:      Harvard FAS RC Holyoke
   source UUID: 1156ed9e-6984-11ea-af52-0201714f6eab
   destination: alcf#dtn_hpss
   destination UUID: bed0e34e-c7fa-4e8c-b291-8f70e634371e
   ```

   Do not substitute a similarly named test collection.
3. Obtain the archive request:
   - one explicit source directory below an approved TPEN output root;
   - completion evidence owned by that experiment's workflow;
   - a new immutable archive ID, such as `run-042-20260820T191500Z`;
   - an approved destination parent, conventionally
     `/home/rhu/globus/TPEN/archives/<archive-id>/`.

The Hooke pair-stability pipeline root currently has no generic root `status.json`; it cannot be an automatic source default. Its future workflow must supply a root-level completion contract before this skill archives it.

## Read-only preflight

Run only these checks first. Keep paths quoted and exact.

```bash
globus whoami

globus ls 1156ed9e-6984-11ea-af52-0201714f6eab:/n/netscratch/kozinsky_lab/Lab/rhu/TPEN

globus ls bed0e34e-c7fa-4e8c-b291-8f70e634371e:/home/rhu
```

Then inspect only the selected source directory and the selected HPSS destination parent. If either consent, mapping, source completion contract, or destination path is unavailable, stop and ask; do not work around it with SSH copies or a different collection.

## Render the transfer before submitting

Set the collection IDs and paths only after the preflight succeeds. The dry run must show `delete_destination_extra: false` and `verify_checksum: true`.

```bash
globus transfer --dry-run --format json \
  --label "TPEN archive <archive-id>" \
  --recursive \
  --sync-level exists \
  --preserve-timestamp \
  --verify-checksum \
  --notify failed,inactive \
  1156ed9e-6984-11ea-af52-0201714f6eab:<source-dir> \
  bed0e34e-c7fa-4e8c-b291-8f70e634371e:/home/rhu/globus/TPEN/archives/<archive-id>/
```

`--sync-level exists` is correct only for a new immutable archive path: it copies missing files without replacing a completed archive. The source directory is placed inside the specified destination parent; include this resulting layout in the task receipt.

For an approved real archive, rerun the same command without `--dry-run`, record the emitted Globus task ID, and retain the exact command settings. Globus creates necessary destination parents during the real submission; do not pre-create them.

## Failure, verification, and recovery

- For a failed transfer, retain the failed task ID and inspect its event/task status before retrying. Use Globus's documented retry path or checksum synchronization only while the source is still immutable. Do not repair a failed transfer with a changed source or destination overwrite.
- Treat `SUCCEEDED` plus Globus checksum verification as transfer integrity evidence, not a restore test.
- Before calling an archive policy operational, perform and record a restore drill of a selected archived run into a new writable staging directory, then verify its manifest/checksums and open a real checkpoint or result artifact.
- Record in Task Orchestrator: source and destination paths, archive ID, collection IDs, command settings, Globus task ID, timestamps, terminal state, and restore evidence.

## Automation boundary

This skill deliberately does not schedule periodic transfers. A recurring archive requires a separate approval for credential/identity lifecycle, source completion registration, alerting, task reconciliation, retention, and recovery ownership. Until then, each archive is a reviewed interactive submission.
