"""Agent 隔离工作区的 Git 时间线和公开文件管理。"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path


_GITIGNORE_CONTENT = """\
sessions/
_engine/
*.nmdb
*.db
*.db-journal
*.db-wal
*.db-shm
__pycache__/
*.pyc
.pytest_cache/
.venv/
"""


class AgentWorkspaceRepository:
    """只管理 Agent 可写工作区，不操作外层项目仓库。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.last_committed_week = 0

    def git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self.path),
            capture_output=True,
            text=True,
            check=check,
        )

    def initialize(self) -> None:
        if (self.path / ".git").exists():
            return
        self.path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main", check=True)
        self.git("config", "user.email", "bash-agent@bossbench.local", check=True)
        self.git("config", "user.name", "BashAgent", check=True)
        gitignore_path = self.path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_GITIGNORE_CONTENT)

    def commit(self, message: str, once_key: str | None = None) -> None:
        if not (self.path / ".git").exists():
            return
        if once_key is not None:
            existing = self.git(
                "log", "--grep", f"[{once_key}]", "--fixed-strings", "--oneline"
            )
            if existing.returncode == 0 and existing.stdout.strip():
                return
            message = f"{message} [{once_key}]"
        self.git("add", "-A", check=True)
        status = self.git("status", "--porcelain", check=True)
        if not status.stdout.strip():
            self.git("commit", "--allow-empty", "-q", "-m", message, check=True)
        else:
            self.git("commit", "-q", "-m", message, check=True)

    def capture_checkpoint_commit(self, day: int) -> str:
        """提交 Agent 当前文件，返回断点精确对应的 Git commit。"""
        if not (self.path / ".git").is_dir():
            raise RuntimeError("Agent workspace is not a Git repository")
        self.git("add", "-A", check=True)
        status = self.git("status", "--porcelain", check=True)
        if status.stdout.strip():
            self.git(
                "commit", "-q", "-m", f"Checkpoint workspace (day {day})", check=True
            )
        head = self.git("rev-parse", "HEAD", check=True).stdout.strip()
        if not head:
            raise RuntimeError("Failed to resolve Agent workspace checkpoint commit")
        return head

    def commit_weeks_up_to(self, sim_day: int) -> None:
        """根据模拟日期补齐不重复的周节点提交。"""
        if sim_day <= 0:
            return
        target_week = sim_day // 7
        while self.last_committed_week < target_week:
            next_week = self.last_committed_week + 1
            self.commit(
                f"Week {next_week} (day {next_week * 7})",
                once_key=f"week-{next_week}",
            )
            # 只有 Git 提交成功后才推进游标，避免静默丢失周节点。
            self.last_committed_week = next_week

    def commit_exists(self, commit: str) -> bool:
        return self.git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0

    def restore_commit(self, commit: str) -> None:
        if not isinstance(commit, str) or not commit:
            raise ValueError("Checkpoint does not contain an Agent workspace commit")
        if not self.commit_exists(commit):
            raise ValueError(f"Agent workspace checkpoint commit does not exist: {commit}")
        # 只回退 Agent 自己的隔离工作区；sessions/ 等忽略目录不会被清理。
        self.git("reset", "--hard", commit, check=True)
        self.git("clean", "-fd", check=True)

    def install_public_artifacts(self, public_dir: Path) -> Path:
        """初始化 Agent 可见文档和公开 CLI，返回 host 侧 CLI 路径。"""
        public_dir = Path(public_dir)
        self.initialize()
        self._copy_docs(public_dir)
        source_cli = public_dir / "novamind-operation"
        if not source_cli.exists():
            raise FileNotFoundError(
                f"{source_cli} does not exist. Did you run "
                "`uv run python scripts/build_public.py`?"
            )
        self._copy_cli(source_cli)
        return source_cli

    def refresh_public_artifacts(self, public_dir: Path) -> None:
        """Git 回退后，使静态文档和 CLI 与当前 host bundle 对齐。"""
        public_dir = Path(public_dir)
        self._copy_docs(public_dir)
        source_cli = public_dir / "novamind-operation"
        if source_cli.exists():
            self._copy_cli(source_cli)

    def _copy_docs(self, public_dir: Path) -> None:
        source = public_dir / "docs"
        destination = self.path / "docs"
        if not source.exists():
            return
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def _copy_cli(self, source: Path) -> None:
        destination = self.path / "novamind-operation"
        shutil.copy2(source, destination)
        destination.chmod(
            destination.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )
