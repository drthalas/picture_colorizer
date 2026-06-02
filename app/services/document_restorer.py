from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time


class DocumentRestorerError(RuntimeError):
    """Raised when a document restoration backend cannot run locally."""


class DocumentRestorer:
    provider: str

    def restore(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenCVDocumentRestorer(DocumentRestorer):
    provider: str = "opencv"

    def restore(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        return Path(input_path)


@dataclass(frozen=True)
class DocResDocumentRestorer(DocumentRestorer):
    provider: str = "docres"
    repo_dir: Path = Path(".external/DocRes")
    python_executable: str = "python"
    task: str = "appearance"
    timeout_seconds: int = 900
    save_dtsprompt: bool = False

    def restore(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve() if output_path is not None else None
        repo_dir = self.repo_dir.resolve()
        self._validate_install(repo_dir)

        start_time = time.time()
        command = [
            self.python_executable,
            "inference.py",
            "--im_path",
            str(input_path),
            "--out_folder",
            str(self._output_folder(repo_dir, output_path)),
            "--task",
            self.task,
            "--save_dtsprompt",
            "1" if self.save_dtsprompt else "0",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise DocumentRestorerError(
                "DocRes inference failed: "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )

        restored_path = self._find_latest_restored_image(repo_dir, input_path, output_path, start_time)
        if output_path is None:
            return restored_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(restored_path, output_path)
        return output_path

    def _validate_install(self, repo_dir: Path) -> None:
        inference_path = repo_dir / "inference.py"
        mbd_path = repo_dir / "data" / "MBD" / "checkpoint" / "mbd.pkl"
        docres_path = repo_dir / "checkpoints" / "docres.pkl"
        missing = [
            str(path)
            for path in (inference_path, mbd_path, docres_path)
            if not path.exists()
        ]
        if missing:
            raise DocumentRestorerError(
                "DocRes backend is not ready. Missing: " + ", ".join(missing)
            )

    def _output_folder(self, repo_dir: Path, output_path: Path | None) -> Path:
        out_folder = output_path.parent if output_path is not None else repo_dir / "restorted"
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder

    def _find_latest_restored_image(
        self,
        repo_dir: Path,
        input_path: Path,
        output_path: Path | None,
        start_time: float,
    ) -> Path:
        restored_dir = self._output_folder(repo_dir, output_path)
        expected_path = restored_dir / f"{input_path.stem}_{self.task}{input_path.suffix}"
        if expected_path.exists() and expected_path.stat().st_mtime >= start_time - 1.0:
            return expected_path

        candidates = [
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")
            for path in restored_dir.glob(pattern)
            if path.stat().st_mtime >= start_time - 1.0
        ]
        if not candidates:
            raise DocumentRestorerError(f"DocRes did not write an image into {restored_dir}")
        return max(candidates, key=lambda path: path.stat().st_mtime)


def create_document_restorer(
    provider: str = "opencv",
    repo_dir: str | Path = ".external/DocRes",
    python_executable: str = "python",
    task: str = "appearance",
    timeout_seconds: int = 900,
    save_dtsprompt: bool = False,
) -> DocumentRestorer:
    normalized = provider.strip().lower()
    if normalized in {"", "opencv", "none", "local"}:
        return OpenCVDocumentRestorer()
    if normalized == "docres":
        return DocResDocumentRestorer(
            repo_dir=Path(repo_dir),
            python_executable=python_executable,
            task=task,
            timeout_seconds=timeout_seconds,
            save_dtsprompt=save_dtsprompt,
        )
    raise DocumentRestorerError(f"Unsupported document restorer provider: {provider}")
