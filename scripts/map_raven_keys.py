"""把"题图 ↔ 正确答案键"配对起来，供离线验证提取器/提示词。

两个数据源，都按"哪一轮 + 第几个 subject"对齐：

A. 新跑的运行：agent 侧 `logs/raven_subject_pairs.jsonl`（解题时落盘：subject 序号、
   题图 sha256、提交的编号与 answer_right），题图另存 `logs/raven_subject_images/<sha12>.png`。
B. 历史运行：agent 日志里 `Agent[<id>] is running Raven subject index N` 给出
   "agent id → subject 序号"，抓图日志 `raven file path ...\\raven_task_data\\<id>_.../`
   给出 "agent id → 题图"（目录名就是 agent id，每次重连换一个 id）。

答案键只存在于服务端：`arena_offline/logs/arena_*.log` 里每个 subject 结算时打印的
`correct_answer`。注意 `eval_res.<时间戳>.json` 存的是**被替换掉的旧记录**，
时间戳是新记录落盘的时刻，按它对齐会错开一条，所以这里只读 arena 日志。

    uv run python scripts/map_raven_keys.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARENA_LOG_DIR = Path(r"D:/Contest/Windows/赛题系统/912/release/release/arena_offline/logs")
RECORD_PATH = Path("logs/raven_subject_pairs.jsonl")
IMAGE_DIR = Path("logs/raven_subject_images")
LEGACY_IMAGE_DIR = Path("logs/raven_task_data")
OUTPUT_PATH = Path("logs/raven_key_pairs.json")

RUN_START_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting subject (\d+)/(\d+)")
KEY_RE = re.compile(r'"correct_answer":\s*(\d+)')
LINE_TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SUBJECT_RE = re.compile(r"Agent\[(\w+)\] is (?:running|solving) Raven subject index (\d+)")
FETCH_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*raven file path .*[\\/]raven_task_data[\\/](\w{10})_"
)


def parse_arena_runs() -> list[dict]:
    """每个 arena 进程 = 一轮；返回 [{start, keys: {subject: key}, name}]。"""
    runs: list[dict] = []
    for path in sorted(glob.glob(str(ARENA_LOG_DIR / "arena_*.log"))):
        start = None
        current = 0
        keys: dict[int, int] = {}
        for line in open(path, encoding="utf-8", errors="replace"):
            stamp = LINE_TIME_RE.match(line)
            when = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S") if stamp else None
            match = RUN_START_RE.search(line)
            if match and when:
                current = int(match.group(2))
                start = start or when
                continue
            key_match = KEY_RE.search(line)
            if key_match and current:
                keys.setdefault(current, int(key_match.group(1)))
        if start and keys:
            runs.append({"start": start, "keys": keys, "name": os.path.basename(path)})
    runs.sort(key=lambda run: run["start"])
    return runs


def run_of(runs: list[dict], when: datetime) -> dict | None:
    found = None
    for run in runs:
        if run["start"] <= when:
            found = run
        else:
            break
    return found


def hash_of_folders() -> tuple[dict[str, str], dict[str, str]]:
    """返回 (目录名前10位 -> hash, hash -> 现存文件路径)。

    目录名是 `<agent_id>_<随机尾巴>`，日志里只记了 agent_id，所以按前 10 位建索引。
    """
    by_prefix: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for path in glob.glob(str(LEGACY_IMAGE_DIR / "*/*.png")):
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        by_prefix.setdefault(os.path.basename(os.path.dirname(path))[:10], digest)
        by_hash.setdefault(digest, path)
    for path in glob.glob(str(IMAGE_DIR / "*.png")):
        by_hash.setdefault(os.path.basename(path)[:12], path)
    return by_prefix, by_hash


def parse_legacy_fetches() -> list[tuple[datetime, int, str | None, str]]:
    """从历史 agent 日志里取 (时间, subject 序号, agent id -> 图 hash, 日志名)。"""
    folder_hash, _ = hash_of_folders()
    rows = []
    for log in sorted(glob.glob("logs/arenaagent_2026-*.log")):
        agent_subject: dict[str, int] = {}
        for line in open(log, encoding="utf-8", errors="replace"):
            match = SUBJECT_RE.search(line)
            if match:
                agent_subject[match.group(1)] = int(match.group(2))
                continue
            fetch = FETCH_RE.search(line)
            if not fetch:
                continue
            when = datetime.strptime(fetch.group(1), "%Y-%m-%d %H:%M:%S")
            agent_id = fetch.group(2)
            subject = agent_subject.get(agent_id)
            if subject is None:
                continue
            rows.append((when, subject, folder_hash.get(agent_id), agent_id))
    return rows


def parse_manifest() -> list[tuple[datetime, int, str | None, str]]:
    rows = []
    if not RECORD_PATH.exists():
        return rows
    for line in RECORD_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") != "subject_image" or not record.get("ts"):
            continue
        rows.append(
            (
                datetime.fromisoformat(record["ts"]),
                int(record.get("subject_index", 0)),
                str(record.get("sha256", ""))[:12],
                str(record.get("agent_id", "")),
            )
        )
    return rows


def main() -> int:
    runs = parse_arena_runs()
    _, hash_paths = hash_of_folders()
    legacies = parse_legacy_fetches()
    manifests = parse_manifest()
    print(f"arena 轮次 {len(runs)}；历史抓图 {len(legacies)}；新格式落盘 {len(manifests)}")

    pairs: dict[str, dict] = {}
    print(f"\n{'时间':<17}{'subject':>8}  {'题图':<13}{'键':<6}来源")
    for when, subject, digest, _ in sorted(legacies + manifests, key=lambda row: row[0]):
        run = run_of(runs, when)
        if run is None:
            continue
        key = run["keys"].get(subject)
        if key is None or not digest:
            continue
        print(f"{when:%m-%d %H:%M:%S}  {subject:>6}  {digest:<13}{key:<6}{run['name'][6:25]}")
        pairs[digest] = {
            "key": key,
            "answers": [int(digit) for digit in str(key)],
            "subject": subject,
            "run": run["name"],
            "image": hash_paths.get(digest, ""),
        }

    existing: dict = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    for digest, item in pairs.items():
        existing.setdefault(digest, item)
        existing[digest].update({k: v for k, v in item.items() if k != "guessed"})
    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== 已知答案的题图 {len(existing)} 张 -> {OUTPUT_PATH} ==")
    for digest, item in sorted(existing.items()):
        where = hash_paths.get(digest) or item.get("image") or "（图不在了）"
        print(f"  {digest} 键 {item['key']}（{item['answers']}）subject {item['subject']} {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
