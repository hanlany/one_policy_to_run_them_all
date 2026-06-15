import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("OPTRA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def artifact_root() -> Path:
    override = os.environ.get("OPTRA_ARTIFACT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_path("artifacts")


def artifact_path(*parts: str) -> Path:
    return artifact_root().joinpath(*parts)
