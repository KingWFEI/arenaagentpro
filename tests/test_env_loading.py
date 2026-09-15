from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arenaagent.builder import load_project_environment


class EnvironmentLoadingTests(unittest.TestCase):
    def test_project_env_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "VLM_CLIENT_CFG_NAME=test-vision-model\n"
                "RAVEN_TEXT_MODEL=deepseek-v4-pro\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch(
                "arenaagent.builder.Path.cwd", return_value=Path(directory)
            ):
                self.assertTrue(load_project_environment())
                self.assertEqual(os.environ["VLM_CLIENT_CFG_NAME"], "test-vision-model")
                self.assertEqual(os.environ["RAVEN_TEXT_MODEL"], "deepseek-v4-pro")

    def test_project_env_does_not_override_explicit_shell_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text(
                "RAVEN_TEXT_MODEL=file-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"RAVEN_TEXT_MODEL": "shell-model"}, clear=True), patch(
                "arenaagent.builder.Path.cwd", return_value=Path(directory)
            ):
                self.assertTrue(load_project_environment())
                self.assertEqual(os.environ["RAVEN_TEXT_MODEL"], "shell-model")


if __name__ == "__main__":
    unittest.main()
