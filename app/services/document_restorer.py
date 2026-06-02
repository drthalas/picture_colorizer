from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    def restore(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        raise DocumentRestorerError(
            "DocRes backend is not installed yet. Keep DOCUMENT_RESTORER_PROVIDER=opencv for now."
        )


def create_document_restorer(provider: str = "opencv") -> DocumentRestorer:
    normalized = provider.strip().lower()
    if normalized in {"", "opencv", "none", "local"}:
        return OpenCVDocumentRestorer()
    if normalized == "docres":
        return DocResDocumentRestorer()
    raise DocumentRestorerError(f"Unsupported document restorer provider: {provider}")
