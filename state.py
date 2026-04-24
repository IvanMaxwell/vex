"""
state.py — Shared state passed between every LangGraph node.
"""
from typing import TypedDict, List


class AgentState(TypedDict):
    # Core task info
    task: str
    plan: List[str]                # Step-by-step plan from Planner
    results: List[str]             # Accumulates results of each step
    errors: List[str]              # Accumulates errors
    current_step: int              # Which plan step Executor is on
    current_agent: str             # Which agent is active right now

    # Conversation / message log (dicts with role+content)
    messages: List[dict]

    # Memory context injected before every agent call
    memory_context: str

    # Task classification (set by Planner, used for routing)
    # Values: "general" | "research" | "ui" | "file" | "system"
    task_type: str

    # Routing flags
    research_needed: bool          # Planner/Executor sets True → routes to Researcher
    ui_needed: bool                # Planner/Executor sets True → routes to UI Controller
    completed: bool                # True → END
    iteration_count: int           # Guards against infinite loops
    retry_count: int               # How many times Planner has been asked to replan
