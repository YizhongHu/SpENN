from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_authoritative_edit.py"
SPEC = importlib.util.spec_from_file_location("check_authoritative_edit", MODULE_PATH)
assert SPEC and SPEC.loader
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)

ITEM_ID = "11111111-1111-4111-8111-111111111111"
ROOT_ID = GUARD.DEFAULT_PROJECT_ROOT_ID


def _run(cwd: Path, *args: str) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _item(
    *,
    item_type: str = "implementation-slice",
    role: str = "work",
    claimed: bool = True,
    body: str = "scope",
) -> dict:
    return {
        "id": ITEM_ID,
        "type": item_type,
        "role": role,
        "isClaimed": claimed,
        "notes": [{"key": "acceptance-contract", "role": "queue", "body": body}],
    }


@contextmanager
def _api(*, item: dict | None = None, root_id: str = ROOT_ID) -> Iterator[str]:
    payload = item or _item()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == f"/api/v1/items/{ITEM_ID}?include=notes":
                body = payload
            elif self.path == f"/api/v1/items/{ITEM_ID}/breadcrumbs":
                body = [{"id": root_id}, {"id": ITEM_ID}]
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class AuthoritativeEditGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        _run(self.repo, "git", "init", "-q", "-b", "codex/guard-test")
        _run(self.repo, "git", "remote", "add", "origin", "https://github.com/YizhongHu/TPEN.git")
        (self.repo / "tracked.txt").write_text("clean\n")
        _run(self.repo, "git", "add", "tracked.txt")
        _run(
            self.repo,
            "git",
            "-c",
            "user.name=Guard Test",
            "-c",
            "user.email=guard@example.test",
            "commit",
            "-qm",
            "fixture",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check(self, api_url: str) -> dict[str, str]:
        return GUARD.check_launch(self.repo, ITEM_ID, api_url, ROOT_ID)

    def test_allows_clean_agent_branch_with_claimed_work_item(self) -> None:
        (self.repo / "untracked-research-data").write_text("preserve me\n")
        with _api() as api_url:
            receipt = self.check(api_url)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["itemId"], ITEM_ID)
        self.assertEqual(receipt["branch"], "codex/guard-test")
        self.assertEqual(receipt["head"], _run(self.repo, "git", "rev-parse", "HEAD"))

    def test_rejects_non_agent_branch(self) -> None:
        _run(self.repo, "git", "branch", "-m", "dev")
        with _api() as api_url, self.assertRaisesRegex(GUARD.GuardFailure, "agent branch"):
            self.check(api_url)

    def test_rejects_dirty_tracked_tree(self) -> None:
        (self.repo / "tracked.txt").write_text("dirty\n")
        with _api() as api_url, self.assertRaisesRegex(GUARD.GuardFailure, "tracked worktree is dirty"):
            self.check(api_url)

    def test_rejects_invalid_item_contract(self) -> None:
        cases = [
            (_item(item_type="research"), "item must have type"),
            (_item(role="queue"), "item must be in work role"),
            (_item(claimed=False), "item must have an active claim"),
            (_item(body="  "), "non-empty queue acceptance-contract"),
        ]
        for item, message in cases:
            with self.subTest(message=message):
                with _api(item=item) as api_url, self.assertRaisesRegex(GUARD.GuardFailure, message):
                    self.check(api_url)

    def test_rejects_item_outside_tpen_root(self) -> None:
        with _api(root_id="22222222-2222-4222-8222-222222222222") as api_url:
            with self.assertRaisesRegex(GUARD.GuardFailure, "must belong to TPEN project root"):
                self.check(api_url)

    def test_rejects_unreachable_server(self) -> None:
        with self.assertRaisesRegex(GUARD.GuardFailure, "Task Orchestrator read failed"):
            self.check("http://127.0.0.1:1")

    def test_rejects_non_loopback_or_unsafe_api_url(self) -> None:
        for api_url in [
            "https://127.0.0.1:3001",
            "http://example.com:3001",
            "http://127.0.0.1:3001/api/v1",
            "http://user:secret@127.0.0.1:3001",
        ]:
            with self.subTest(api_url=api_url):
                with self.assertRaisesRegex(GUARD.GuardFailure, "Task Orchestrator API"):
                    self.check(api_url)


if __name__ == "__main__":
    unittest.main()
