"""Guard: every config the training entrypoint needs must be committed.

`wandb/` in .gitignore is unanchored, so it also matched
molplatte/src/configs/wandb/ and silently untracked that config group. Because
config.yaml's defaults list requires `- wandb: default`, a fresh clone could not
compose its Hydra config at all -- every run failed, with or without logging.

It went unnoticed for a long time because every run happened in a working tree
where the file exists on disk. Only a clone reveals it, so this test asks git
what is tracked rather than asking the filesystem what exists.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "molplatte" / "src" / "configs"


def tracked(path: Path) -> bool:
    r = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(path)],
        capture_output=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
class TestConfigsAreCommitted:
    def test_root_config_is_tracked(self):
        assert tracked(CONFIGS / "config.yaml")

    def test_every_defaults_group_is_tracked(self):
        """Each `- <group>: <name>` in the defaults list must exist AND be committed."""
        text = (CONFIGS / "config.yaml").read_text()
        block = text.split("defaults:", 1)[1].split("\n\n", 1)[0]
        groups = re.findall(r"^\s*-\s*([a-z_]+)\s*:\s*([a-z_]+)\s*$", block, re.M)
        assert groups, "could not parse the defaults list"
        missing = [
            f"{g}/{n}.yaml"
            for g, n in groups
            if g != "_self_" and not tracked(CONFIGS / g / f"{n}.yaml")
        ]
        assert not missing, (
            f"config groups required by config.yaml but NOT committed: {missing}. "
            "A fresh clone will fail Hydra composition before any run starts."
        )

    def test_no_ignore_pattern_matches_a_config_path(self):
        """No ignore pattern may MATCH a config path, tracked or not.

        --no-index is essential. Without it `git check-ignore` skips tracked
        files, so once the offending file is committed the check silently
        passes while the pattern is still there -- ready to swallow the next
        config group somebody adds.
        """
        swallowed = [
            str(p.relative_to(REPO))
            for p in CONFIGS.rglob("*.yaml")
            if subprocess.run(
                ["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", str(p)]
            ).returncode == 0
        ]
        assert not swallowed, (
            f"an ignore pattern matches these config files: {swallowed}. "
            "They are safe only while they stay tracked; the next config group "
            "added under that path would be silently untracked."
        )
