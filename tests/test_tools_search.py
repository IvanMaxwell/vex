import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from tools import _format_search_results
from tools import _serpapi_search
from tools import web_search


class WebSearchTests(unittest.TestCase):
    def test_serpapi_search_requires_api_key(self):
        with patch("tools.SERPAPI_KEY", ""):
            with self.assertRaises(RuntimeError) as ctx:
                _serpapi_search("Apple stock price", 3)

        self.assertIn("SERPAPI_KEY", str(ctx.exception))

    def test_serpapi_search_parses_google_response(self):
        payload = {
            "organic_results": [
                {
                    "title": "Apple Inc. (AAPL) Stock Price",
                    "link": "https://finance.example.com/aapl",
                    "displayed_link": "finance.example.com",
                    "snippet": "AAPL price and chart.",
                }
            ]
        }

        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with patch("tools.SERPAPI_KEY", "test-key"):
            with patch("tools.requests.get", return_value=response):
                results = _serpapi_search("Apple stock price", 3)

        self.assertEqual(results[0]["title"], "Apple Inc. (AAPL) Stock Price")
        self.assertEqual(results[0]["domain"], "finance.example.com")

    def test_web_search_uses_serpapi_results(self):
        class Allow:
            def request(self, *args, **kwargs):
                return True

        with patch("tools.permission_manager", Allow()):
            with patch(
                "tools._serpapi_search",
                return_value=[
                    {
                        "title": "Apple Stock Price",
                        "href": "https://finance.example.com/aapl",
                        "domain": "finance.example.com",
                        "body": "AAPL 123.45 USD",
                    }
                ],
            ) as mock_search:
                result = web_search("Check Apple stock price", 5)

        self.assertIn("Apple Stock Price", result)
        mock_search.assert_called()

    def test_web_search_surfaces_serpapi_failures(self):
        class Allow:
            def request(self, *args, **kwargs):
                return True

        with patch("tools.permission_manager", Allow()):
            with patch(
                "tools._serpapi_search",
                side_effect=RuntimeError("invalid API key"),
            ):
                result = web_search("Check Apple stock price", 5)

        self.assertIn("ERROR: Web search failed.", result)
        self.assertIn("invalid API key", result)

    def test_format_search_results_includes_domain(self):
        text = _format_search_results(
            [
                {
                    "title": "Example",
                    "href": "https://example.com/page",
                    "domain": "example.com",
                    "body": "Snippet here",
                }
            ]
        )
        self.assertIn("Source: example.com", text)


if __name__ == "__main__":
    unittest.main()
