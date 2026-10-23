from typing_extensions import Annotated, Sequence, TypedDict, Any

import operator
from dataclasses import dataclass, field


import json


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


class AgentState(TypedDict):
    messages: Annotated[Sequence[Any], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]


@dataclass
class DayState:

    pre: dict = field(default_factory=dict)
    proc: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)

    def put(self, phase: str, agent: str, payload: Any, *, ticker: str | None = None) -> None:
        getattr(self, phase)[(agent, ticker) if ticker else agent] = payload

    def get(self, phase: str, agent: str | None = None, *, ticker: str | None = None) -> Any:
        bucket = getattr(self, phase)
        if agent is None:
            return dict(bucket)
        return bucket.get((agent, ticker) if ticker else agent)


def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)

    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2, ensure_ascii=False))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(output)

    print("=" * 48)
