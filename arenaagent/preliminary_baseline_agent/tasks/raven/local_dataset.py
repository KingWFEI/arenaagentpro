from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from arenaagent.vlm_agent.raven_skill import normalize_raven_image_list, resolve_raven_image_path


def _correct_triple(evaluation: Any) -> list[int] | None:
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get("correct_answer")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        digits = value
    else:
        text = str(value or "").strip()
        if len(text) != 3 or not text.isdigit():
            return None
        digits = [int(char) for char in text]
    try:
        normalized = [int(float(item)) for item in digits]
    except (TypeError, ValueError):
        return None
    return normalized if all(1 <= item <= 8 for item in normalized) else None


def collect_labeled_raven_subject(agent: Any, evaluation: Any) -> Path | None:
    """Persist a train-mode Raven canvas as three supervised local samples."""
    answers = _correct_triple(evaluation)
    source = resolve_raven_image_path(str(getattr(agent, "_raven_image_temp_path", "") or ""))
    if answers is None or source is None:
        return None
    groups = normalize_raven_image_list(source)
    if not groups or len(groups) != 3:
        return None

    image_bytes = Path(source).read_bytes()
    fingerprint = hashlib.sha256(image_bytes).hexdigest()
    log_dir = Path(str(getattr(getattr(agent, "cfg", None), "log_dir", "logs") or "logs"))
    data_dir = Path(os.getenv("RAVEN_LOCAL_DATASET_DIR", "").strip() or log_dir / "raven_local_dataset")
    subject_dir = data_dir / fingerprint[:16]
    subject_dir.mkdir(parents=True, exist_ok=True)
    canvas_path = subject_dir / "canvas.png"
    if not canvas_path.exists():
        Image.open(source).convert("RGB").save(canvas_path)

    arrays = np.stack(
        [
            np.stack(
                [np.asarray(panel.convert("L").resize((224, 224), Image.Resampling.BILINEAR)) for panel in group]
            )
            for group in groups
        ]
    ).astype(np.uint8)
    sample_path = subject_dir / "questions.npz"
    np.savez_compressed(sample_path, images=arrays, answers=np.asarray(answers, dtype=np.int64))
    metadata = {
        "fingerprint": fingerprint,
        "answers": answers,
        "canvas": str(canvas_path.resolve()),
        "samples": str(sample_path.resolve()),
    }
    (subject_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Collected labeled Raven subject {} answers={} at {}", fingerprint[:16], answers, subject_dir)
    return subject_dir
