"""离线对照：不起仿真，直接拿存下来的题图跑**生产路径**（三题并行、每题矩阵+候选两张图）。

    uv run python scripts/probe_raven_prompt.py [题图路径] [第几次]

题图默认取日志里那张（arena subject 1 的复现图，键 582 = 5/8/2）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arenaagent.builder import load_project_environment  # noqa: E402
from arenaagent.preliminary_baseline_agent.tasks.raven.pure_vision import solve_pure_vision  # noqa: E402
from arenaagent.preliminary_baseline_agent.tasks.raven.vision_client import (  # noqa: E402
    build_raven_vision_client_from_env,
)

DEFAULT_IMAGE = "logs/raven_task_data/62f357be81_82lyvig5/task_data_001.png"
FALLBACK_EXPECTED = {1: 5, 2: 8, 3: 2}  # subject 1（键 582），仅在题图不在配对表里时使用
KEYS_PATH = Path("logs/raven_key_pairs.json")


def expected_of(path: str) -> dict[int, int]:
    """按题图 hash 从配对表里取真实答案；取不到就退回 subject 1 的 582。"""
    if not KEYS_PATH.exists():
        return dict(FALLBACK_EXPECTED)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    try:
        pairs = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return dict(FALLBACK_EXPECTED)
    item = pairs.get(digest)
    if item and item.get("answers"):
        return {index: int(answer) for index, answer in enumerate(item["answers"], start=1)}
    return dict(FALLBACK_EXPECTED)


def run_single(path: str, trial: str) -> int:
    expected = expected_of(path)
    load_project_environment()
    client = build_raven_vision_client_from_env()
    if client is None:
        print("K3 client unavailable")
        return 1

    try:
        attempt = max(1, int(trial))
    except ValueError:
        attempt = 1
    answers, diagnostics = solve_pure_vision(client, path, attempt=attempt)
    print(f"用时 {diagnostics.get('elapsed_seconds')}s，发送 {diagnostics.get('sent_kilobytes')}KB，图块 {diagnostics.get('panel_sizes')}")
    hits = 0
    for item in diagnostics.get("per_question", []):
        index = item.get("question")
        got = item.get("answer")
        hits += bool(got == expected.get(index))
        print(
            f"Q{index} 期望{expected.get(index)} 得到{got} "
            f"({item.get('seconds')}s, 矩阵{item.get('matrix_size')} 候选{item.get('option_size')}) "
            f"-> {str(item.get('reason'))[:110]}"
        )
    print(f"== 第{trial}次：{hits}/3（答案 {answers}）==")
    return 0


def run_all() -> int:
    """把配对表里的每张图都跑一遍，输出逐题对错 + 总计。"""
    if not KEYS_PATH.exists():
        print("没有 logs/raven_key_pairs.json，先跑 scripts/map_raven_keys.py")
        return 1
    pairs = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    load_project_environment()
    client = build_raven_vision_client_from_env()
    if client is None:
        print("K3 client unavailable")
        return 1
    per_question: dict[int, list[bool]] = {1: [], 2: [], 3: []}
    subject_hits = 0
    for digest, item in sorted(pairs.items()):
        path = item.get("image") or ""
        if not path or not Path(path).exists():
            print(f"  跳过 {digest}（题图不在本地）")
            continue
        expected = {i: int(a) for i, a in enumerate(item["answers"], start=1)}
        answers, diagnostics = solve_pure_vision(client, path)
        got = answers or []
        marks = []
        for index in (1, 2, 3):
            ok = len(got) >= index and got[index - 1] == expected[index]
            per_question[index].append(ok)
            marks.append(f"Q{index}:{got[index - 1] if len(got) >= index else '-'}期望{expected[index]}{'对' if ok else '错'}")
        all_right = len(got) == 3 and all(len(got) >= i and got[i - 1] == expected[i] for i in (1, 2, 3))
        subject_hits += bool(all_right)
        print(f"  {digest} 键{item['answers']} {' '.join(marks)} 三题{'全对' if all_right else '未全对'}")
    total = sum(len(v) for v in per_question.values())
    hits = sum(sum(v) for v in per_question.values())
    print("\n== 逐题命中率 ==")
    for index in (1, 2, 3):
        marks = per_question[index]
        print(f"  Q{index}: {sum(marks)}/{len(marks)}")
    print(f"  合计 {hits}/{total}；三题全对 {subject_hits}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        raise SystemExit(run_all())
    raise SystemExit(run_single(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE, sys.argv[2] if len(sys.argv) > 2 else "1"))
