from __future__ import annotations

import unittest
from pathlib import Path

from nefas.preprocessing import intermediate_workspace


class PreprocessingTests(unittest.TestCase):
    def test_intermediate_workspace_uses_config_file_name(self) -> None:
        workspace = intermediate_workspace(
            Path("configs/united_states.yaml"),
            Path("outputs"),
        )

        self.assertEqual(workspace, Path("outputs/united_states"))


if __name__ == "__main__":
    unittest.main()
