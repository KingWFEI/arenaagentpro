"""出一份完整的"slot ↔ 题图 ↔ 答案键"配对表（跑完一轮采集后用）。

数据：`logs/raven_subject_pairs.jsonl`（agent 落盘）+ 最新那份 arena 日志里的
`correct_answer`。输出同时打印并写到 `logs/raven_pair_table.md`。

    uv run python scripts/raven_pair_report.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARENA_LOG_DIR = Path(r"D:/Contest/Windows/赛题系统/912/release/release/arena_offline/logs")
RECORD_PATH = Path("logs/raven_subject_pairs.jsonl")
OUTPUT_PATH = Path("logs/raven_pair_table.md")
IMAGE_DIR = Path("logs/raven_subject_images")

RUN_START_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting subject (\d+)/(\d+)")
KEY_RE = re.compile(r'"correct_answer":\s*(\d+)')


def latest_run_keys() -> tuple[str, dict[int, int]]:
    """取最新一份 arena 日志的 slot -> 键（一轮 = 一个进程）。"""
    log = max(glob.glob(str(ARENA_LOG_DIR / "arena_*.log")), key=os.path.getmtime)
    current = 0
    keys: dict[int, int] = {}
    for line in open(log, encoding="utf-8", errors="replace"):
        match = RUN_START_RE.search(line)
        if match:
            current = int(match.group(2))
            continue
        key_match = KEY_RE.search(line)
        if key_match and current:
            keys.setdefault(current, int(key_match.group(1)))
    return os.path.basename(log), keys


def load_records() -> tuple[dict[int, str], dict[int, list[tuple[int, list, object]]]]:
    images: dict[int, str] = {}
    submits: dict[int, list[tuple[int, list, object]]] = defaultdict(list)
    if not RECORD_PATH.exists():
        return images, submits
    for line in RECORD_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        subject = record.get("subject_index")
        if not isinstance(subject, int):
            continue
        if record.get("event") == "subject_image":
            images[subject] = str(record.get("sha256", ""))[:12]
        elif record.get("event") == "submit":
            submits[subject].append(
                (int(record.get("attempt") or 0), record.get("answer"), record.get("answer_right"))
            )
    for rows in submits.values():
        rows.sort()
    return images, submits


def main() -> int:
    run_name, keys = latest_run_keys()
    images, submits = load_records()
    lines = [f"# 瑞文 slot ↔ 题图 ↔ 答案键 配对表", ""]
    lines.append(f"数据来源：`{run_name}`（arena 日志）+ `logs/raven_subject_pairs.jsonl`（agent 落盘）")
    lines.append("")
    lines.append("| slot | 题图 (sha256 前12位) | 键 | 正确答案 | 我们的提交（按顺序） | 结果 |")
    lines.append("|---|---|---|---|---|---|")

    solved, attempted = [], []
    resolved: dict[int, object] = {}
    for subject in range(1, 11):
        digest = images.get(subject, "（未采集）")
        rows = submits.get(subject, [])
        right = any(result for _, _, result in rows)
        key = keys.get(subject)
        inferred = ""
        if key is None:
            # 赛题端在"agent 已断开"时会跳过评测、连 correct_answer 都不写。
            # 这种情况下用"被接受的那个答案"反推键（服务端判对才会接受）。
            accepted = next((answer for _, answer, result in rows if result), None)
            if isinstance(accepted, list):
                key = int("".join(str(digit) for digit in accepted))
                inferred = "（推断）"
        expect = "、".join(str(digit) for digit in str(key)) if key is not None else "（未采集到）"
        guesses = " → ".join(
            f"[{''.join(str(d) for d in answer)}]" if isinstance(answer, list) else str(answer)
            for _, answer, _ in rows
        ) or "未提交"
        mark = "✓ 答对" if right else ("✗ 全错" if rows else "— 未作答")
        if rows and not keys.get(subject):
            mark += "（赛题端跳过评分 ⚠）" if right else ""
        lines.append(
            f"| {subject} | `{digest}` | {key if key is not None else '?'}{inferred} | "
            f"{expect} | {guesses} | {mark} |"
        )
        if key is not None:
            resolved[subject] = key
        if right:
            solved.append(subject)
        elif rows:
            attempted.append(subject)

    by_digest: dict[str, list[int]] = defaultdict(list)
    for subject, digest in images.items():
        by_digest[digest].append(subject)
    lines += ["", "## 去重后：本地题库只有 6 道题", "", "| 题图 | 答案键 | 占用的 slot |", "|---|---|---|"]
    for digest, slots in sorted(by_digest.items(), key=lambda item: item[1][0]):
        slot_keys = {resolved.get(slot) for slot in slots} - {None}
        same = "（同图同键 ✓）" if len(slot_keys) == 1 else f"（键不一致 ✗ {sorted(slot_keys)}）"
        lines.append(f"| `{digest}` | {sorted(slot_keys)} | {slots} {same} |")

    total_questions = len(by_digest) * 3
    lines += [
        "",
        "## 统计",
        "",
        f"- 不同题图：**{len(by_digest)} 张**，共 **{total_questions} 道题**",
        f"- 答对的 slot：{solved or '无'}；提交过但全错的 slot：{attempted or '无'}",
        f"- 题图副本：`{IMAGE_DIR}`（不会被临时目录清理）",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"已写入 {OUTPUT_PATH}")

    # 同步给离线探针用：题图 hash -> 真实答案（含上面推断出来的键）
    pairs_path = Path("logs/raven_key_pairs.json")
    existing: dict = {}
    if pairs_path.exists():
        try:
            existing = json.loads(pairs_path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    for subject, digest in images.items():
        key = resolved.get(subject)
        if key is None or digest == "（未采集）":
            continue
        path = IMAGE_DIR / f"{digest}.png"
        if not path.exists():
            fallback = [
                item for item in glob.glob(str(Path("logs/raven_task_data") / "*" / "*.png"))
                if os.path.basename(os.path.dirname(item))[:10] in digest[:10]
            ]
            path = Path(fallback[0]) if fallback else path
        existing[digest] = {
            "key": int(key),
            "answers": [int(digit) for digit in str(key)],
            "subject": subject,
            "image": str(path) if path.exists() else "",
        }
    pairs_path.write_text(json.dumps(existing, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已同步 {len(existing)} 条答案到 {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
