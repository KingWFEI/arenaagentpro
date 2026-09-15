from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.vlm_agent.skills.raven import (
    crop_group_image_to_pil_groups,
    crop_group_image_to_subplots,
    solve_raven,
)


def _fail(agent: Any, error: str) -> dict[str, Any]:
    fail_result = getattr(agent, "_fail_result", None)
    if callable(fail_result):
        return fail_result(error=error)
    return {"result": "failed", "error": str(error)}


def run_raven_inference(image_list: list[list[Image.Image]], structure: list[Any]) -> list[list[int]] | None:
    """Run the Raven model and normalize its predictions."""
    argv_backup = sys.argv[:]
    try:
        # solve_raven internally calls argparse.parse_args(), isolate from process args.
        sys.argv = [argv_backup[0]] if argv_backup else [""]
        prediction = solve_raven(image_list=image_list, structure=structure)
    except Exception as exc:
        logger.warning("solve_raven api call failed: {}", exc)
        return None
    finally:
        sys.argv = argv_backup

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    if not isinstance(prediction, list) or not prediction:
        return None

    normalized: list[list[int]] = []
    for triple in prediction:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        try:
            normalized.append([int(triple[0]), int(triple[1]), int(triple[2])])
        except Exception:
            continue
    return normalized or None


def get_raven_ranked_candidates(
    agent: Any,
    cache_key: str,
    image_list: list[list[Image.Image]],
    structure: list[Any],
    legacy_candidates: list[list[int]] | None = None,
    rejected_answers: list[list[int]] | None = None,
    previous_attempts: list[dict[str, Any]] | None = None,
) -> list[list[int]] | None:
    cache = getattr(agent, "_raven_candidates_cache", None)
    if not isinstance(cache, dict):
        return None

    legacy_candidates = legacy_candidates or []
    candidates: list[list[int]] = []
    try:
        # Import lazily so non-Raven tasks never initialize the hybrid CV/rule stack.
        from arenaagent.preliminary_baseline_agent.tasks.raven.solver import HybridRavenSolver

        result = HybridRavenSolver(
            vlm_client=getattr(agent, "vlm_client", None),
            text_client=getattr(agent, "raven_text_client", None),
        ).solve(
            image_groups=image_list,
            legacy_ranked=legacy_candidates,
            rejected_triples=rejected_answers,
            previous_attempts=previous_attempts,
        )
        candidates = result.ranked_triples
        agent._raven_last_diagnostics = result.diagnostics
    except Exception as exc:
        logger.exception("Hybrid Raven solver failed; use legacy ranking if available: {}", exc)
        candidates = legacy_candidates
    if not candidates:
        return None

    cache[cache_key] = candidates
    return candidates


def _raven_reasoning_history(agent: Any, cache_key: str) -> list[dict[str, Any]]:
    histories = getattr(agent, "_raven_reasoning_histories", None)
    if not isinstance(histories, dict):
        histories = {}
        agent._raven_reasoning_histories = histories
    return histories.setdefault(cache_key, [])


def _compact_attempts(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in history[-4:]:
        diagnostics = item.get("diagnostics") or {}
        compact.append(
            {
                "attempt": item.get("attempt"),
                "answer": item.get("answer"),
                "status": item.get("status"),
                "visual_votes": diagnostics.get("visual_votes", []),
                "text_votes": diagnostics.get("text_votes", []),
                "question_confidences": diagnostics.get("question_confidences", []),
                "question_margins": diagnostics.get("question_margins", []),
                "question_probabilities": diagnostics.get("question_probabilities", []),
                "rule_top_candidates": diagnostics.get("rule_top_candidates", []),
                "legacy_top_candidates": diagnostics.get("legacy_top_candidates", []),
                "reasoning_mode": diagnostics.get("reasoning_mode"),
                "revisited_questions": diagnostics.get("revisited_questions", []),
            }
        )
    return compact


def _persist_raven_reasoning_trace(agent: Any, cache_key: str, history: list[dict[str, Any]]) -> None:
    cfg = getattr(agent, "cfg", None)
    log_dir = getattr(cfg, "log_dir", None)
    if not log_dir:
        return
    try:
        trace_dir = os.path.join(str(log_dir), "raven_reasoning")
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"{getattr(agent, 'agent_id', 'agent')}_{cache_key[:16]}.json")
        with open(path, "w", encoding="utf-8") as writer:
            json.dump(
                {
                    "image_sha256": cache_key,
                    "note": "Auditable model outputs and rule summaries; no hidden chain-of-thought is recorded.",
                    "attempts": history,
                },
                writer,
                ensure_ascii=False,
                indent=2,
            )
        agent._raven_reasoning_trace_path = path
    except Exception as exc:
        logger.warning("Failed to persist Raven reasoning trace: {}", exc)


def _normalized_evaluation_answer(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        answer = [int(float(item)) for item in value]
    except (TypeError, ValueError):
        return None
    return answer if all(1 <= item <= 8 for item in answer) else None


def _evaluation_confirms_success(evaluation: dict[str, Any], answer: list[int] | None) -> bool:
    is_right = evaluation.get("is_right")
    if is_right is True or str(is_right).strip().lower() == "true":
        return True
    correct_answer = evaluation.get("correct_answer")
    if answer is None or correct_answer is None:
        return False
    try:
        return int(correct_answer) == int("".join(str(item) for item in answer))
    except (TypeError, ValueError):
        return False


def record_confirmed_raven_experience(agent: Any, evaluation: Any) -> bool:
    """Persist techniques only after the task service explicitly confirms success."""
    if not isinstance(evaluation, dict):
        return False
    evaluation_answer = _normalized_evaluation_answer(evaluation.get("answer"))
    histories = getattr(agent, "_raven_reasoning_histories", None)
    if not isinstance(histories, dict):
        return False

    selected_key = ""
    selected_history: list[dict[str, Any]] | None = None
    selected_attempt: dict[str, Any] | None = None
    for cache_key, history in reversed(list(histories.items())):
        if not isinstance(history, list):
            continue
        for attempt in reversed(history):
            candidate = _normalized_evaluation_answer(attempt.get("answer"))
            if candidate is None:
                continue
            if evaluation_answer is not None and candidate != evaluation_answer:
                continue
            if attempt.get("status") != "pending":
                continue
            selected_key = str(cache_key)
            selected_history = history
            selected_attempt = attempt
            evaluation_answer = evaluation_answer or candidate
            break
        if selected_attempt is not None:
            break
    if selected_attempt is None or selected_history is None:
        return False
    if not _evaluation_confirms_success(evaluation, evaluation_answer):
        return False

    resolved_image_path = resolve_raven_image_path(getattr(agent, "_raven_image_temp_path", ""))
    image_groups = normalize_raven_image_list(resolved_image_path) if resolved_image_path else None
    if not image_groups or evaluation_answer is None:
        logger.warning("Raven success was confirmed but its source image is unavailable for experience learning")
        return False

    try:
        from arenaagent.preliminary_baseline_agent.tasks.raven.experience import (
            default_experience_path,
            record_successful_subject,
        )

        log_dir = str(getattr(getattr(agent, "cfg", None), "log_dir", "logs") or "logs")
        learned = record_successful_subject(
            image_groups,
            evaluation_answer,
            diagnostics=dict(selected_attempt.get("diagnostics") or {}),
            subject_fingerprint=selected_key,
            path=default_experience_path(log_dir),
        )
    except Exception as exc:
        logger.warning("Failed to record confirmed Raven experience: {}", exc)
        return False

    selected_attempt["status"] = "accepted"
    _persist_raven_reasoning_trace(agent, selected_key, selected_history)
    return learned


def get_raven_legacy_candidates(
    agent: Any,
    cache_key: str,
    image_list: list[list[Image.Image]],
    structure: list[Any],
) -> list[list[int]]:
    cache = getattr(agent, "_raven_legacy_candidates_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        agent._raven_legacy_candidates_cache = cache
    if cache_key not in cache:
        cache[cache_key] = run_raven_inference(image_list=image_list, structure=structure) or []
    return cache[cache_key]


def normalize_raven_image_list(raw_image_list: Any) -> list[list[Image.Image]] | None:
    # New input format: one whole Raven canvas image (base64/path/bytes/PIL).
    if isinstance(raw_image_list, (str, bytes, Image.Image)):
        return group_image_to_raven_list(raw_image_list)

    if not isinstance(raw_image_list, list):
        return None

    if len(raw_image_list) == 1 and isinstance(raw_image_list[0], (str, bytes, Image.Image)):
        return group_image_to_raven_list(raw_image_list[0])

    # A: [[16 imgs], [16 imgs], [16 imgs]]
    if raw_image_list and isinstance(raw_image_list[0], list):
        groups: list[list[Image.Image]] = []
        for group in raw_image_list:
            pil_group = to_pil_group(group)
            if not pil_group:
                return None
            groups.append(pil_group)
        return groups if groups else None

    pil_images = to_pil_group(raw_image_list)
    if not pil_images:
        return None

    # C: [48 imgs] -> split into 3 groups
    if len(pil_images) >= 48:
        return [pil_images[0:16], pil_images[16:32], pil_images[32:48]]

    return None


def group_image_to_raven_list(group_image: Any) -> list[list[Image.Image]] | None:
    source_image = to_pil_image(group_image)
    if source_image is None:
        return None

    try:
        groups = crop_group_image_to_pil_groups(source_image)
    except Exception as exc:
        logger.warning("in-memory Raven canvas crop failed: {}", exc)
        return None

    # Keep disk crops as an opt-in diagnostic only. Normal inference never saves 48 files.
    if os.getenv("RAVEN_SAVE_CROPS", "").strip().lower() in {"1", "true", "yes", "on"}:
        image_path = group_image if isinstance(group_image, str) and os.path.exists(group_image) else None
        image_path = image_path or materialize_group_image(source_image)
        if image_path:
            crop_group_image_to_subplots(image_path)
    return [[panel.convert("L") for panel in group] for group in groups]


def materialize_group_image(value: Any) -> str | None:
    out_dir = "/tmp/raven_input_images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"group_{int(time.time() * 1000)}_{os.getpid()}.png")

    try:
        if isinstance(value, Image.Image):
            value.save(out_path)
            return out_path

        if isinstance(value, bytes):
            img = Image.open(io.BytesIO(value)).convert("L")
            img.save(out_path)
            return out_path

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            if text.startswith("data:image"):
                _, b64_data = text.split(",", 1)
                raw = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(raw)).convert("L")
                img.save(out_path)
                return out_path

            if os.path.exists(text):
                return text

            # Fallback: treat as raw base64 without data URL prefix.
            raw = base64.b64decode(text)
            img = Image.open(io.BytesIO(raw)).convert("L")
            img.save(out_path)
            return out_path
    except Exception as exc:
        logger.warning("Failed to materialize Raven group image: {}", exc)
        return None

    return None


def to_pil_group(values: Any) -> list[Image.Image] | None:
    if not isinstance(values, list):
        return None
    images: list[Image.Image] = []
    for value in values:
        image = to_pil_image(value)
        if image is None:
            return None
        images.append(image)
    return images


def to_pil_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("L")

    if isinstance(value, bytes):
        try:
            return Image.open(io.BytesIO(value)).convert("L")
        except Exception:
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.startswith("data:image"):
            try:
                _, b64_data = text.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("L")
            except Exception:
                return None

        if os.path.exists(text):
            try:
                return Image.open(text).convert("L")
            except Exception:
                return None

        image_bytes, _ = decode_base64_image(text)
        if image_bytes is not None:
            try:
                return Image.open(io.BytesIO(image_bytes)).convert("L")
            except Exception:
                return None

    return None


def build_raven_attempt_key(image_list: list[list[Image.Image]], structure: list[Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(structure, ensure_ascii=False, default=str).encode("utf-8"))
    for group in image_list:
        for image in group:
            hasher.update(image.mode.encode("utf-8"))
            hasher.update(str(image.size).encode("utf-8"))
            hasher.update(image.tobytes())
    return hasher.hexdigest()


def materialize_task_data_images(agent: Any, task_data: Any) -> None:
    cleanup_raven_temp_images(agent)

    created_paths: list[str] = []
    temp_dir: str | None = None
    image_index = 0

    def ensure_temp_dir() -> str:
        nonlocal temp_dir
        if temp_dir is None:
            base_dir = os.path.join(getattr(getattr(agent, "cfg", None), "log_dir", "") or "logs", "raven_task_data")
            os.makedirs(base_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix=f"{getattr(agent, 'agent_id', 'agent')}_", dir=base_dir)
        return temp_dir

    def convert(value: Any) -> Any:
        nonlocal image_index

        if isinstance(value, dict):
            for item in value.values():
                convert(item)
            return value

        if isinstance(value, list):
            for item in value:
                convert(item)
            return value

        if not isinstance(value, str):
            return value

        image_bytes, extension = decode_base64_image(value)
        if image_bytes is None:
            return value

        image_index += 1
        file_path = os.path.join(ensure_temp_dir(), f"task_data_{image_index:03d}{extension}")
        with open(file_path, "wb") as writer:
            writer.write(image_bytes)
        logger.debug("raven file path {}", file_path)
        created_paths.append(file_path)
        return file_path

    convert(task_data)
    if not created_paths:
        logger.debug("no raven image path extracted from task data")
        return

    agent._raven_image_temp_path = created_paths[0] if len(created_paths) == 1 else (temp_dir or created_paths[0])


def cleanup_raven_temp_images(agent: Any) -> None:
    temp_path = getattr(agent, "_raven_image_temp_path", "")
    if not temp_path:
        return

    agent._raven_image_temp_path = ""
    try:
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
            return
        if os.path.isfile(temp_path):
            parent_dir = os.path.dirname(temp_path)
            os.remove(temp_path)
            if parent_dir and os.path.isdir(parent_dir):
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass
    except Exception as exc:
        logger.warning("Failed to cleanup Raven temp images {}: {}", temp_path, exc)


def decode_base64_image(value: str) -> tuple[bytes | None, str]:
    text = value.strip()
    if not text or os.path.exists(text):
        return None, ""

    header = ""
    payload = text
    if text.startswith("data:image"):
        parts = text.split(",", 1)
        if len(parts) != 2:
            return None, ""
        header, payload = parts
    elif len(text) < 128:
        return None, ""

    try:
        image_bytes = base64.b64decode(payload, validate=not bool(header))
    except (binascii.Error, ValueError):
        return None, ""

    extension = guess_image_extension(image_bytes, header)
    if not extension:
        return None, ""
    return image_bytes, extension


def guess_image_extension(image_bytes: bytes, header: str = "") -> str:
    lower_header = header.lower()
    if "image/png" in lower_header:
        return ".png"
    if "image/jpeg" in lower_header or "image/jpg" in lower_header:
        return ".jpg"
    if "image/webp" in lower_header:
        return ".webp"
    if "image/gif" in lower_header:
        return ".gif"
    if "image/bmp" in lower_header:
        return ".bmp"
    if "image/svg+xml" in lower_header:
        return ".svg"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.lstrip().startswith(b"<svg"):
        return ".svg"
    return ""


def resolve_raven_image_path(image_temp_path: str) -> str | None:
    if not image_temp_path:
        return None
    if os.path.isfile(image_temp_path):
        return image_temp_path
    if not os.path.isdir(image_temp_path):
        return None

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    for root, _, files in os.walk(image_temp_path):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            if os.path.splitext(file_name)[1].lower() in image_extensions:
                return file_path
    return None


def handle(agent: Any, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Handle the solve_raven action for agents with prepared Raven task data."""
    image_temp_path = getattr(agent, "_raven_image_temp_path", "")
    resolved_image_path = resolve_raven_image_path(image_temp_path)
    if not resolved_image_path:
        logger.warning("raven image path is not exist")
        return _fail(agent, "raven image path not exist")

    get_param = getattr(agent, "_get_param", None)
    if not callable(get_param):
        return _fail(agent, "raven skill helper missing: _get_param")

    structure = get_param(params, "structure", default=[])
    if not isinstance(structure, list):
        structure = []

    logger.debug("solve_raven using image path {}", resolved_image_path)
    image_list = normalize_raven_image_list(resolved_image_path)
    if not image_list:
        logger.warning("invalid parameter: image_list")
        return _fail(agent, "invalid parameter: image_list")

    cache_key = build_raven_attempt_key(image_list, structure)
    attempted_cache = getattr(agent, "_raven_attempted_answers", None)
    if not isinstance(attempted_cache, dict):
        attempted_cache = {}
        agent._raven_attempted_answers = attempted_cache
    attempted = attempted_cache.setdefault(cache_key, set())
    history = _raven_reasoning_history(agent, cache_key)
    if history and history[-1].get("status") == "pending":
        history[-1]["status"] = "rejected"
        # Persist the server feedback before any potentially slow model call.
        _persist_raven_reasoning_trace(agent, cache_key, history)

    legacy_candidates = get_raven_legacy_candidates(agent, cache_key, image_list, structure)
    phase = "legacy_fast_path"
    ranked_candidates = legacy_candidates
    diagnostics: dict[str, Any] = {}
    if attempted or not legacy_candidates:
        phase = "reasoned_correction"
        rejected_answers = [list(item) for item in sorted(attempted)]
        ranked_candidates = get_raven_ranked_candidates(
            agent,
            cache_key,
            image_list,
            structure,
            legacy_candidates=legacy_candidates,
            rejected_answers=rejected_answers,
            previous_attempts=_compact_attempts(history),
        )
        diagnostics = getattr(agent, "_raven_last_diagnostics", {})
        if diagnostics.get("reasoning_mode") == "targeted_parallel_revision":
            phase = "targeted_llm_revision"
        if not ranked_candidates:
            return _fail(agent, "Raven reasoned correction unavailable; refusing blind enumeration")
    chosen_answer = ranked_candidates[0] if ranked_candidates else None
    if chosen_answer is not None and tuple(chosen_answer) in attempted:
        chosen_answer = None
    if chosen_answer is None:
        logger.warning("Raven produced no new evidence-backed answer; refusing candidate enumeration")
        return _fail(agent, "Raven produced no new evidence-backed answer; refusing candidate enumeration")
    attempted.add(tuple(chosen_answer))
    history.append(
        {
            "attempt": len(history) + 1,
            "phase": phase,
            "answer": chosen_answer,
            "status": "pending",
            "rejected_before": [
                list(item) for item in sorted(attempted) if list(item) != chosen_answer
            ],
            "diagnostics": diagnostics,
        }
    )
    _persist_raven_reasoning_trace(agent, cache_key, history)

    # Only the task server can declare a Raven subject finished. A wrong answer
    # leaves it RUNNING; setting the local flag here would deadlock both sides.
    logger.info(
        "Raven {} answer attempt {} selected {} (reasoning trace={})",
        phase,
        len(attempted),
        chosen_answer,
        getattr(agent, "_raven_reasoning_trace_path", "memory"),
    )
    action_space = getattr(agent, "action_space", {}) or {}
    key = action_space.get("key") or "action"
    return {key: chosen_answer}
