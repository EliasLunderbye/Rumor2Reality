"""Compatibility imports for the Gradio interface's shared verification service."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.service import DEFAULT_MODEL, pipeline_events, run_pipeline

__all__ = ["DEFAULT_MODEL", "pipeline_events", "run_pipeline"]
