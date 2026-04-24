"""
graph.py — LangGraph StateGraph wiring the four agents together.

Flow:
  planner ─┬→ researcher ──┐
           ├→ ui_controller─┤
           └→ executor   ←─┘
                  │
                  ├→ researcher → executor
                  ├→ ui_controller → executor
                  ├→ planner (error / replan)
                  └→ END
"""
from langgraph.graph import StateGraph, END

from state import AgentState
from agents import planner_node, executor_node, researcher_node, ui_controller_node
from runtime import streamer
from config import MAX_ITERATIONS, MAX_RETRIES


# ══════════════════════════════════════════════════════════════
#  ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def route_after_planner(state: AgentState) -> str:
    """Route from Planner based on task-type classification set by the Planner node."""
    if state.get("research_needed", False):
        streamer.status(
            f"🔍 Task type '{state.get('task_type')}' — routing directly to Researcher.",
            "System",
        )
        return "researcher"
    if state.get("ui_needed", False):
        streamer.status(
            f"🖥️ Task type '{state.get('task_type')}' — routing directly to UI Controller.",
            "System",
        )
        return "ui_controller"
    return "executor"


def route_after_executor(state: AgentState) -> str:
    if state.get("completed", False):
        streamer.status("🏁 Task complete — pipeline finished.", "System")
        return "end"

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        streamer.status("⚠️ Max iterations reached — stopping.", "System")
        return "end"

    # Errors and retries left → replan
    if state.get("errors"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            streamer.status("🔄 Errors detected — asking Planner to replan.", "System")
            return "planner"
        else:
            streamer.status("❌ Max retries reached with errors — stopping.", "System")
            return "end"

    if state.get("research_needed", False):
        return "researcher"

    if state.get("ui_needed", False):
        return "ui_controller"

    # Still have plan steps left → keep executing
    if state.get("current_step", 0) < len(state.get("plan", [])):
        return "executor"

    return "end"


def route_after_researcher(state: AgentState) -> str:
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"
    return "executor"


def route_after_ui(state: AgentState) -> str:
    if state.get("completed", False):
        streamer.status("🏁 Task complete — pipeline finished.", "System")
        return "end"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        streamer.status("⚠️ Max iterations reached — stopping.", "System")
        return "end"
        
    if state.get("errors"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            streamer.status("🔄 Errors detected — asking Planner to replan.", "System")
            return "planner"
        else:
            streamer.status("❌ Max retries reached with errors — stopping.", "System")
            return "end"

    # UI tasks are terminal — never bounce back to executor.
    # The UI Controller handles all steps for a UI-type task.
    if state.get("task_type") == "ui":
        streamer.status("🏁 UI task complete.", "System")
        return "end"
    if state.get("current_step", 0) < len(state.get("plan", [])):
        return "executor"
    return "end"


# ══════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner",       planner_node)
    g.add_node("executor",      executor_node)
    g.add_node("researcher",    researcher_node)
    g.add_node("ui_controller", ui_controller_node)

    g.set_entry_point("planner")

    # Planner routes based on task-type classification
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor":      "executor",
            "researcher":    "researcher",
            "ui_controller": "ui_controller",
        },
    )

    # Executor routes conditionally
    g.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor":      "executor",
            "researcher":    "researcher",
            "ui_controller": "ui_controller",
            "planner":       "planner",
            "end":           END,
        },
    )

    # Researcher routes back to Executor
    g.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {"executor": "executor", "end": END},
    )

    # UI Controller routes back to Executor or ends
    g.add_conditional_edges(
        "ui_controller",
        route_after_ui,
        {"executor": "executor", "end": END},
    )

    return g.compile()


# ══════════════════════════════════════════════════════════════
#  INITIAL STATE FACTORY
# ══════════════════════════════════════════════════════════════
def create_initial_state(task: str, memory_context: str) -> AgentState:
    return AgentState(
        task=task,
        plan=[],
        results=[],
        errors=[],
        current_step=0,
        current_agent="",
        messages=[],
        memory_context=memory_context,
        task_type="general",
        research_needed=False,
        ui_needed=False,
        completed=False,
        iteration_count=0,
        retry_count=0,
    )
