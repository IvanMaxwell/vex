import unittest
from unittest.mock import patch

from agents import (
    _clean_llm_output,
    _parse_tool_calls,
    executor_node,
    planner_node,
    ui_controller_node,
)
from tools import _search_queries


class AgentLogicTests(unittest.TestCase):
    def test_search_queries_rewrites_plain_check_prefix(self):
        queries = _search_queries("Check Apple stock price")
        self.assertEqual(queries[0], "Check Apple stock price")
        self.assertIn("Apple stock price", queries)
        self.assertIn("Apple stock price today", queries)

    def test_clean_llm_output_removes_think_and_stop_tokens(self):
        raw = "<think>hidden</think>\nhello<|endoftext|><|im_start|>"
        self.assertEqual(_clean_llm_output(raw), "hello")

    def test_parse_tool_call_tag_format(self):
        text = """
<tool_call>
{"name": "search", "arguments": {"query": "Apple stock price today"}}
</tool_call>
"""
        self.assertEqual(
            _parse_tool_calls(text),
            [{"tool": "web_search", "params": {"query": "Apple stock price today"}}],
        )

    @patch("agents.streamer")
    @patch("agents.dispatch_tool")
    @patch("agents._stream_llm", return_value="")
    def test_executor_delegates_research_when_empty_reply_implies_web_lookup(
        self,
        _mock_llm,
        mock_dispatch,
        mock_streamer,
    ):
        state = {
            "task": "Check Apple stock price",
            "plan": [
                "Use web_search to get the latest quote from a reputable finance source."
            ],
            "results": [],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = executor_node(state)

        self.assertFalse(updated["completed"])
        self.assertTrue(updated["research_needed"])
        self.assertFalse(updated["ui_needed"])
        self.assertEqual(updated["current_step"], 0)
        self.assertIn("Research needed:", updated["results"][-1])
        mock_dispatch.assert_not_called()
        mock_streamer.info.assert_called()

    @patch("agents.streamer")
    @patch("agents.dispatch_tool")
    @patch(
        "agents._stream_llm",
        return_value="""TOOL_CALL: web_search
PARAMS: {"query": "Apple stock price today", "max_results": 5}
END_TOOL_CALL""",
    )
    def test_executor_delegates_explicit_web_search_to_researcher(
        self,
        _mock_llm,
        mock_dispatch,
        _mock_streamer,
    ):
        state = {
            "task": "Check Apple stock price",
            "plan": [
                "Use web_search to get the latest quote from a reputable finance source."
            ],
            "results": [],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = executor_node(state)

        self.assertTrue(updated["research_needed"])
        self.assertEqual(updated["errors"], [])
        self.assertIn("Apple stock price today", updated["results"][-1])
        mock_dispatch.assert_not_called()

    @patch("agents.streamer")
    @patch("agents.dispatch_tool")
    @patch(
        "agents._stream_llm",
        return_value="""TOOL_CALL: ui_control
PARAMS: {"action": "open_whatsapp_send", "target": "amma", "text": "hi"}
END_TOOL_CALL""",
    )
    def test_executor_delegates_ui_tool_calls_to_ui_controller(
        self,
        _mock_llm,
        mock_dispatch,
        _mock_streamer,
    ):
        state = {
            "task": 'Open WhatsApp, search "amma", and type hi to amma',
            "plan": [
                'Open WhatsApp, search "amma", and type hi to amma using the desktop UI.'
            ],
            "results": [],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = executor_node(state)

        self.assertTrue(updated["ui_needed"])
        self.assertFalse(updated["research_needed"])
        self.assertEqual(updated["current_step"], 0)
        self.assertIn("UI action needed:", updated["results"][-1])
        mock_dispatch.assert_not_called()

    @patch("agents.streamer")
    @patch("agents.dispatch_tool", return_value="WhatsApp opened. Typed 'hi' in chat with 'amma'. Press Enter to send.")
    @patch("agents._stream_llm", return_value="")
    def test_ui_controller_uses_whatsapp_fallback_when_model_returns_nothing(
        self,
        _mock_llm,
        mock_dispatch,
        _mock_streamer,
    ):
        state = {
            "task": 'Open WhatsApp, search "amma", and type hi to amma',
            "plan": ['Open WhatsApp, search "amma", and type hi to amma using the desktop UI.'],
            "results": ['UI action needed: open whatsapp and type hi to amma'],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": True,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = ui_controller_node(state)

        mock_dispatch.assert_called_once_with(
            "ui_control",
            {"action": "open_whatsapp_send", "target": "amma", "text": "hi"},
        )
        self.assertFalse(updated["ui_needed"])
        self.assertIn("[UI] open_whatsapp_send:", updated["results"][-1])

    @patch("agents.streamer")
    @patch(
        "agents._stream_llm",
        return_value='1. Search and Open WhatsApp\n2. Search "amma"\n3. Type "hi" to amma',
    )
    def test_planner_heuristically_classifies_whatsapp_task_as_ui_when_task_type_missing(
        self,
        _mock_llm,
        _mock_streamer,
    ):
        state = {
            "task": 'search and Open WhatsApp , search "amma "and type hi to amma',
            "plan": [],
            "results": [],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = planner_node(state)

        self.assertEqual(updated["task_type"], "ui")
        self.assertTrue(updated["ui_needed"])
        self.assertFalse(updated["research_needed"])

    @patch("agents.streamer")
    @patch("agents.dispatch_tool")
    @patch("agents._stream_llm", return_value="")
    def test_executor_prefers_ui_handoff_over_web_search_for_whatsapp_task(
        self,
        _mock_llm,
        mock_dispatch,
        _mock_streamer,
    ):
        state = {
            "task": 'search and Open WhatsApp , search "amma "and type hi to amma',
            "plan": ['Search and Open WhatsApp'],
            "results": [],
            "errors": [],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = executor_node(state)

        self.assertTrue(updated["ui_needed"])
        self.assertFalse(updated["research_needed"])
        self.assertIn("UI action needed:", updated["results"][-1])
        mock_dispatch.assert_not_called()

    @patch("agents.streamer")
    def test_planner_clears_active_errors_on_replan(self, _mock_streamer):
        state = {
            "task": "Check CPU usage",
            "plan": [],
            "results": [],
            "errors": ["Step 1: previous failure"],
            "current_step": 0,
            "current_agent": "",
            "messages": [],
            "memory_context": "No previous context.",
            "research_needed": False,
            "ui_needed": False,
            "completed": False,
            "iteration_count": 0,
            "retry_count": 0,
        }

        updated = planner_node(state)

        self.assertEqual(updated["errors"], [])
        self.assertEqual(updated["retry_count"], 1)


if __name__ == "__main__":
    unittest.main()
