import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from process_floorplan import check_room_geometry


class RoomGeometryGateTest(unittest.TestCase):
    def test_only_a_reviewable_mask_can_continue_after_an_error(self):
        mask_error = "topology_mask_change_unconfirmed"
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            path = Path(root) / "quality.json"
            for code, errors, expected_exit in (
                    (0, [], None), (2, [mask_error], None), (2, [], 2),
                    (2, [mask_error, "invalid_polygon"], 2), (1, [mask_error], 1)):
                path.write_text(json.dumps({"issues": [{"code": e, "severity": "error"} for e in errors]}))
                with self.subTest(code=code, errors=errors), patch(
                        "process_floorplan.subprocess.run", return_value=subprocess.CompletedProcess([], code)):
                    if expected_exit is None:
                        check_room_geometry(["validator"], path)
                    else:
                        with self.assertRaises(SystemExit) as stopped:
                            check_room_geometry(["validator"], path)
                        self.assertEqual(stopped.exception.code, expected_exit)
