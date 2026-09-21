from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def hf_token() -> str | None:
    """Read a local Hugging Face token without writing credentials to this repo."""
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
    return os.getenv("HF_TOKEN")
