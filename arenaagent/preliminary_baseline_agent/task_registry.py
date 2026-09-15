from __future__ import annotations

from arenaagent.preliminary_baseline_agent.tasks.base import (
    GenericTaskStrategy,
    TaskStrategy,
)
from arenaagent.preliminary_baseline_agent.tasks.counting.strategy import CountingStrategy
from arenaagent.preliminary_baseline_agent.tasks.jigsaw.strategy import JigsawStrategy
from arenaagent.preliminary_baseline_agent.tasks.npc.strategy import NpcStrategy
from arenaagent.preliminary_baseline_agent.tasks.raven.strategy import RavenStrategy
from arenaagent.preliminary_baseline_agent.tasks.tidyroom.strategy import TidyRoomStrategy

TASK_STRATEGIES: dict[str, type[TaskStrategy]] = {
    strategy.task_type: strategy
    for strategy in (TidyRoomStrategy, JigsawStrategy, CountingStrategy, NpcStrategy, RavenStrategy)
}


def create_task_strategy(task_type: str) -> TaskStrategy:
    strategy_class = TASK_STRATEGIES.get(task_type)
    if strategy_class is None:
        return GenericTaskStrategy(task_type)
    return strategy_class()


def supported_task_types() -> tuple[str, ...]:
    return tuple(TASK_STRATEGIES)
