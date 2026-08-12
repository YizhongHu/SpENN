#!/usr/bin/env python3
"""Fail-closed launch check for authoritative TPEN repository edits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn, Sequence

DEFAULT_API_URL = "http://127.0.0.1:3001"
DEFAULT_PROJECT_ROOT_ID = "58348558-a4c8-4ed7-8f7b-829ef8163145"
EXPECTED_REMOTE_PATH = "/YizhongHu/TPEN"
EXPECTED_ITEM_TYPE = "implementation-slice"


class GuardFailure(RuntimeError):
    """An authoritative-edit precondition was not satisfied."""


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise GuardFailure(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _validated_api_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise GuardFailure("Task Orchestrator API must use plain HTTP on a loopback host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GuardFailure("Task Orchestrator API URL must not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise GuardFailure("Task Orchestrator API URL must not contain a path")
    try:
        port = parsed.port
    except ValueError as error:
        raise GuardFailure(f"Task Orchestrator API URL has an invalid port: {error}") from error
    if port is None:
        raise GuardFailure("Task Orchestrator API URL must include an explicit port")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"http://{host}:{port}"


def _read_json(api_url: str, path: str) -> Any:
    request = urllib.request.Request(f"{api_url}{path}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise GuardFailure(f"Task Orchestrator read failed for {path}: {error}") from error


def _validate_repository(cwd: Path) -> tuple[Path, str, str]:
    top_level = Path(_git(cwd, "rev-parse", "--show-toplevel")).resolve()
    remote = urllib.parse.urlparse(_git(top_level, "remote", "get-url", "origin"))
    remote_path = remote.path.removesuffix(".git")
    if remote.hostname != "github.com" or remote_path != EXPECTED_REMOTE_PATH:
        raise GuardFailure("cwd is not a worktree of github.com/YizhongHu/TPEN")

    branch = _git(top_level, "branch", "--show-current")
    if not branch.startswith(("codex/", "claude/")):
        raise GuardFailure("authoritative edits require an agent branch: codex/** or claude/**")

    dirty = _git(top_level, "status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise GuardFailure("tracked worktree is dirty; commit or restore tracked changes before editing")

    return top_level, branch, _git(top_level, "rev-parse", "HEAD")


def _validate_item(api_url: str, item_id: str, project_root_id: str) -> None:
    quoted_id = urllib.parse.quote(item_id, safe="")
    item = _read_json(api_url, f"/api/v1/items/{quoted_id}?include=notes")

    if item.get("type") != EXPECTED_ITEM_TYPE:
        raise GuardFailure(f"item must have type {EXPECTED_ITEM_TYPE!r}")
    if item.get("role") != "work":
        raise GuardFailure("item must be in work role")
    if item.get("isClaimed") is not True:
        raise GuardFailure("item must have an active claim")

    breadcrumbs = _read_json(api_url, f"/api/v1/items/{quoted_id}/breadcrumbs")
    if not breadcrumbs or breadcrumbs[0].get("id") != project_root_id:
        raise GuardFailure(f"item must belong to TPEN project root {project_root_id}")

    acceptance = next(
        (note for note in item.get("notes") or [] if note.get("key") == "acceptance-contract"),
        None,
    )
    if not acceptance or acceptance.get("role") != "queue" or not acceptance.get("body", "").strip():
        raise GuardFailure("item must have a non-empty queue acceptance-contract note")


def check_launch(cwd: Path, item_id: str, api_url: str, project_root_id: str) -> dict[str, str]:
    """Validate the repository lane and Task Orchestrator work contract."""

    validated_url = _validated_api_url(api_url)
    top_level, branch, head = _validate_repository(cwd.resolve())
    _validate_item(validated_url, item_id, project_root_id)
    return {
        "status": "ok",
        "itemId": item_id,
        "projectRootId": project_root_id,
        "cwd": str(top_level),
        "branch": branch,
        "head": head,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True, help="Task Orchestrator item UUID or hex prefix")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="TPEN worktree to validate")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Loopback Task Orchestrator REST URL")
    parser.add_argument(
        "--project-root-id",
        default=DEFAULT_PROJECT_ROOT_ID,
        help="Expected TPEN project root UUID",
    )
    return parser


def _fail(message: str) -> NoReturn:
    print(json.dumps({"status": "blocked", "reason": message}, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = check_launch(args.cwd, args.item, args.api_url, args.project_root_id)
    except GuardFailure as error:
        _fail(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
