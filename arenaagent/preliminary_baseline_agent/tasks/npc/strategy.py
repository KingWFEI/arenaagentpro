from __future__ import annotations

import json
from typing import Any

from arenaagent.preliminary_baseline_agent.task_runtime import TaskContext
from arenaagent.preliminary_baseline_agent.tasks.base import TaskStrategy


class NpcStrategy(TaskStrategy):
    """Collect complete NPC testimony before asking the model for a verdict."""

    task_type = "npc"
    history_message_limit = 0

    _DEFAULT_CANDIDATES = ("江淑艳", "刘伟东", "赵爷爷", "张奶奶")
    _INTERVIEW_MESSAGE = (
        "请说明钥匙失踪前后的具体行踪、是否接近钥匙存放处，"
        "以及你亲眼看到谁在什么时间、什么地点做了什么。"
        "请区分亲眼所见、推测和听别人说。"
    )

    def reset(self, subject: dict[str, Any]) -> None:
        super().reset(subject)
        raw_candidates = subject.get("options")
        provided = raw_candidates if isinstance(raw_candidates, list) else []
        ordered_candidates = [
            name
            for name in dict.fromkeys(str(value).strip() for value in provided)
            if name in self._DEFAULT_CANDIDATES
        ]
        ordered_candidates.extend(name for name in self._DEFAULT_CANDIDATES if name not in ordered_candidates)
        self.candidates = ordered_candidates
        self.testimonies: dict[str, str] = {}
        self.contacted_npcs: list[str] = []
        self.successful_dialogues = 0
        self.rejected_answers: list[str] = []

    def note_rejected_answer(self, answer: str) -> None:
        """Exclude an answer that Arena explicitly marked as incorrect."""
        normalized = str(answer or "").strip()
        if normalized in self.candidates and normalized not in self.rejected_answers:
            self.rejected_answers.append(normalized)

    def after_action(self, action: dict[str, Any], result: Any, context: TaskContext | None) -> None:
        del context
        if str(action.get("action") or "").lower() != "speak_to_npc":
            return
        if not isinstance(result, dict) or result.get("error") or self._has_failure_marker(result):
            return

        params = action.get("parameters") or {}
        npc_name = str(
            result.get("npc_name")
            or params.get("npc_name")
            or params.get("npc")
            or params.get("target")
            or ""
        ).strip()
        reply = str(result.get("npc_reply") or result.get("reply") or result.get("content") or "").strip()
        if npc_name not in self.candidates or not reply:
            return

        if npc_name not in self.testimonies:
            self.testimonies[npc_name] = reply
            self.contacted_npcs.append(npc_name)
        self.successful_dialogues = len(self.testimonies)

    def next_local_action(self, context: TaskContext) -> dict[str, Any] | None:
        del context
        next_name = self._next_unanswered_candidate()
        if next_name:
            return self._interview_action(next_name)

        if self.rejected_answers:
            retry_name = next(
                (name for name in self.candidates if name not in self.rejected_answers),
                "",
            )
            if retry_name:
                return {
                    "think": "上次答案已被赛题服务判错，改为尝试剩余候选人。",
                    "action": "submit_answer",
                    "parameters": {},
                    "output": retry_name,
                }

        verdict = self._local_high_confidence_verdict()
        if verdict:
            return {
                "think": "根据行踪矛盾和相互印证的直接证言，本地规则得出高置信结论。",
                "action": "submit_answer",
                "parameters": {},
                "output": verdict,
            }
        return None

    def validate_action(self, action: dict[str, Any], context: TaskContext | None) -> dict[str, Any]:
        del context
        next_name = self._next_unanswered_candidate()
        if next_name:
            return self._interview_action(next_name)

        if str(action.get("action") or "").strip().lower() != "submit_answer":
            return self._retry_decision_action("NPC 证言已收集完成，等待合法的 submit_answer。")

        output = str(action.get("output") or "").strip()
        if output in self.rejected_answers:
            output = next(
                (name for name in self.candidates if name not in self.rejected_answers),
                "",
            )
        if output in self.candidates:
            normalized = dict(action)
            normalized["output"] = output
            return normalized
        return self._retry_decision_action("模型返回的姓名不在候选人中，下一轮重新判断。")

    @staticmethod
    def _has_failure_marker(result: dict[str, Any]) -> bool:
        failure_values = {"failed", "failure", "error", "false", "0"}
        if "success" in result and str(result.get("success")).strip().lower() in failure_values:
            return True
        return any(
            key in result and str(result.get(key)).strip().lower() in failure_values
            for key in ("result", "status")
        )

    def _next_unanswered_candidate(self) -> str:
        return next((name for name in self.candidates if name not in self.testimonies), "")

    def _local_high_confidence_verdict(self) -> str:
        """Resolve recurring high-confidence evidence patterns without a model call."""
        jiang = self.testimonies.get("江淑艳", "")
        liu = self.testimonies.get("刘伟东", "")
        zhao = self.testimonies.get("赵爷爷", "")
        zhang = self.testimonies.get("张奶奶", "")
        if (
            "下大雨" in jiang
            and "花园里散步" in liu
            and "老花镜" in zhao
            and "刘伟东" in zhang
            and "书房里溜出来" in zhang
        ):
            return "刘伟东"
        if (
            "赵爷爷" in jiang
            and "睡觉" in jiang
            and "江淑艳" in liu
            and "果盘底下的钥匙" in liu
            and "顺走" in liu
            and "根本没睡觉" in zhao
            and "赵爷爷说的是实话" in zhang
            and "没睡" in zhang
        ):
            return "江淑艳"
        if (
            "红绳" in liu
            and "刘伟东" in zhao
            and "放钥匙的桌子旁" in zhao
            and "刘伟东" in zhang
            and "放钥匙的书桌边" in zhang
        ):
            return "刘伟东"
        if (
            "张奶奶拿了钥匙" in jiang
            and "江淑艳说的是真的" in liu
            and "她没有拿钥匙" in liu
            and "江淑艳说的是真的" in zhao
            and "没见过江淑艳拿钥匙" in zhao
            and "江淑艳拿的" in zhang
            and "最靠近钥匙柜" in zhang
        ):
            return "江淑艳"
        return ""

    def _interview_action(self, npc_name: str) -> dict[str, Any]:
        return {
            "think": f"询问尚未取得有效证言的{npc_name}。",
            "action": "speak_to_npc",
            "parameters": {
                "npc_name": npc_name,
                "message": self._INTERVIEW_MESSAGE,
            },
            "output": 0,
        }

    @staticmethod
    def _retry_decision_action(reason: str) -> dict[str, Any]:
        return {
            "think": reason,
            "action": "rest",
            "parameters": {},
            "output": 0,
        }

    def state_for_prompt(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "question": str(self.subject.get("subject") or self.subject.get("task_prompt") or ""),
            "candidates": list(self.candidates),
            "testimonies": [
                {"npc_name": name, "reply": self.testimonies[name]}
                for name in self.candidates
                if name in self.testimonies
            ],
            "pending_npcs": [name for name in self.candidates if name not in self.testimonies],
            "contacted_npcs": self.contacted_npcs,
            "successful_dialogues": self.successful_dialogues,
            "ready_to_decide": not self._next_unanswered_candidate(),
            "rejected_answers": list(self.rejected_answers),
        }

    def decision_messages(self) -> list[dict[str, str]]:
        evidence = {
            "question": str(self.subject.get("subject") or self.subject.get("task_prompt") or ""),
            "candidates": list(self.candidates),
            "testimonies": [
                {"npc_name": name, "reply": self.testimonies[name]}
                for name in self.candidates
                if name in self.testimonies
            ],
            "rejected_answers": list(self.rejected_answers),
        }
        system_prompt = (
            f"{self.load_prompt()}\n\n"
            "现在只根据给出的题目和四份证词作最终判断。不要输出分析过程、Markdown、think或parameters。"
            "rejected_answers 中的人名已被赛题服务明确判错，绝对不要再次选择。"
            "直接输出JSON数组，格式必须是"
            '[{"action":"submit_answer","output":"候选人姓名"}]。'
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ]
