import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from demo import app as api
from demo.service_info import format_updated_at, load_service_info


class ServiceInfoTests(unittest.TestCase):
    def test_git_identity_uses_commit_date_and_moscow_display(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".git").mkdir()
            result = subprocess.CompletedProcess([], 0, "abcdef1\n2026-11-13T08:10:00+00:00\n", "")
            with patch("demo.service_info.subprocess.run", return_value=result):
                info = load_service_info(root, {})
        self.assertEqual(info["version"], "0.1.0+abcdef1")
        self.assertEqual(info["revision"], "abcdef1")
        self.assertEqual(format_updated_at(info["updated_at"]), "13.11.2026 11:10 МСК")

    def test_exported_build_uses_overrides_and_unknown_date_is_not_start_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("demo.service_info.subprocess.run") as git:
                self.assertIsNone(load_service_info(root, {})["updated_at"])
                git.assert_not_called()
            info = load_service_info(root, {"FLAT_RECO_SERVICE_VERSION": "1.2.3",
                                           "FLAT_RECO_UPDATED_AT": "2026-11-13T08:10:00Z"})
            self.assertEqual(info["version"], "1.2.3")
            self.assertEqual(info["updated_at"], "2026-11-13T08:10:00+00:00")
            (root / ".git").mkdir()
            with patch("demo.service_info.subprocess.run", side_effect=FileNotFoundError):
                self.assertIsNone(load_service_info(root, {})["revision"])
            for value in ("bad-date", "2026-11-13T08:10:00", ""):
                self.assertEqual(format_updated_at(value), "не указана")


class ServiceInfoPageTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_and_api_share_build_identity_with_escaped_html(self):
        info = {"version": '1.2.3+<build>', "revision": None,
                "updated_at": "2026-11-13T08:10:00+00:00"}
        with patch.object(api, "SERVICE_INFO", info):
            page = await api.index()
            response = await api.service_version()
        html = page.body.decode()
        self.assertIn("1.2.3+&lt;build&gt;", html)
        self.assertIn("13.11.2026 11:10 МСК", html)
        self.assertNotIn("{{SERVICE_", html)
        self.assertEqual(json.loads(response.body), info)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
