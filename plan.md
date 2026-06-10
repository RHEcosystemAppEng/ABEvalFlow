# Harbor Local Environment Implementation Plan

## Branch
```bash
cd /Users/gziv/Dev/skills_eval_corrections
git checkout -b feature/local-environment || git checkout feature/local-environment
```

## Step 1: Add enum value

**File:** `src/harbor/models/environment_type.py`

Add this line after `USE_COMPUTER = "use-computer"`:
```python
    LOCAL = "local"
```

## Step 2: Create LocalEnvironment class

**File:** `src/harbor/environments/local.py` (NEW FILE)

```python
"""Local environment that runs commands directly via subprocess."""

import asyncio
import shutil
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType


class LocalEnvironment(BaseEnvironment):
    """Environment that runs commands locally without containers.
    
    Useful for A2A evaluation where the agent is external and
    the verifier can run directly in the host process.
    """

    @staticmethod
    def type() -> str:
        return EnvironmentType.LOCAL

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            mounted=True,
            disable_internet=False,
            gpus=False,
            tpus=False,
            windows=False,
            docker_compose=False,
            network_allowlist=False,
            dynamic_network_policy=False,
        )

    def _validate_definition(self) -> None:
        """No validation needed - no Dockerfile required."""
        pass

    async def start(self, force_build: bool = False) -> None:
        """No-op - no container to start."""
        self.logger.debug("LocalEnvironment: start (no-op)")

    async def stop(self, delete: bool = False) -> None:
        """No-op - no container to stop."""
        self.logger.debug("LocalEnvironment: stop (no-op)")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute command via subprocess."""
        merged_env = self._merge_env(env)
        
        # Build environment dict for subprocess
        import os
        proc_env = os.environ.copy()
        if merged_env:
            proc_env.update(merged_env)

        self.logger.debug(f"LocalEnvironment exec: {command}")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=proc_env,
            )
            
            if timeout_sec:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_sec,
                )
            else:
                stdout, stderr = await proc.communicate()
            
            return ExecResult(
                stdout=stdout.decode() if stdout else None,
                stderr=stderr.decode() if stderr else None,
                return_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(
                stdout=None,
                stderr="Command timed out",
                return_code=124,
            )
        except Exception as e:
            return ExecResult(
                stdout=None,
                stderr=str(e),
                return_code=1,
            )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Copy file from source to target."""
        src = Path(source_path)
        dst = Path(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.logger.debug(f"LocalEnvironment: copied {src} -> {dst}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Copy directory from source to target."""
        src = Path(source_dir)
        dst = Path(target_dir)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        self.logger.debug(f"LocalEnvironment: copied dir {src} -> {dst}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Copy file from environment to local."""
        src = Path(source_path)
        dst = Path(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.logger.debug(f"LocalEnvironment: downloaded {src} -> {dst}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Copy directory from environment to local."""
        src = Path(source_dir)
        dst = Path(target_dir)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        self.logger.debug(f"LocalEnvironment: downloaded dir {src} -> {dst}")
```

## Step 3: Register in factory

**File:** `src/harbor/environments/factory.py`

Add this entry to `_ENVIRONMENT_REGISTRY` dict (after the OPENSHIFT entry):
```python
    EnvironmentType.LOCAL: _EnvEntry(
        "harbor.environments.local",
        "LocalEnvironment",
        None,
    ),
```

## Step 4: Test locally

```bash
cd /Users/gziv/Dev/ABEvalFlow
pip install -e /Users/gziv/Dev/skills_eval_corrections

# Test with a simple task
harbor run -e local -c test-config.yaml
```

## Step 5: Commit and push

```bash
cd /Users/gziv/Dev/skills_eval_corrections
git add -A
git commit -m "feat: add local environment type for direct subprocess execution"
git push -u origin feature/local-environment
```

## Step 6: Update ABEvalFlow evaluate.yaml

In the A2A Harbor config generation section, change environment type from `openshift` to `local`:
```python
"environment": {
    "type": "local",  # Changed from openshift
}
```
