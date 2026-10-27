"""Entrypoint, packaging and resource path contracts."""
from pathlib import Path
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from perch import cli
from perch.settings import ROOT, MODEL_PATHS


class EntrypointTests(unittest.TestCase):
    def test_camera_and_video_delegate_to_the_same_controller(self):
        for entry, source in ((cli.camera_main, "camera"), (cli.video_main, "video")):
            argv = ["--headless", "--max-frames", "1"]
            with self.subTest(source=source), patch("perch.app.run") as run:
                self.assertEqual(entry(argv), 0)
                self.assertIs(run.call_args.args[0], argv)
                args = run.call_args.kwargs["parser"].parse_args([])
                self.assertEqual(args.source, source)
                self.assertEqual((args.burst_frames, args.min_samples), (3, 3))
                self.assertEqual(args.profile, "detail")

    def test_entrypoints_reject_the_other_input_source(self):
        for entry, wrong in ((cli.camera_main, "video"), (cli.video_main, "camera")):
            with self.subTest(source=wrong), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    entry(["--source", wrong])
                self.assertEqual(error.exception.code, 2)

    def test_video_size_alias_selects_the_export_profile(self):
        for size, profile in ((640, "fast"), (960, "detail")):
            args = cli.build_video_parser().parse_args(["--imgsz", str(size)])
            self.assertEqual(args.profile, profile)

    def test_defaults_resolve_to_standalone_resources(self):
        args = cli.build_video_parser().parse_args([])
        for path in (args.left, args.right, args.calib, *MODEL_PATHS.values()):
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertTrue(path.exists(), path)
        self.assertEqual(args.snapshot_dir, ROOT / "outputs/video")
        for directory in MODEL_PATHS.values():
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["source_weights"], "best-seg.pt")

    def test_launchers_work_from_another_working_directory(self):
        for name in ("perch_detector.py", "perch_video_seg.py"):
            result = subprocess.run([sys.executable, str(ROOT / name), "--help"],
                                    cwd=ROOT.parent, capture_output=True, text=True, check=True)
            self.assertIn("--profile", result.stdout)
            self.assertIn("--burst-frames", result.stdout)

    def test_runtime_errors_are_reported_without_a_traceback(self):
        with patch("perch.app.run", side_effect=RuntimeError("Camera unavailable")):
            with contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(cli.camera_main([]), 1)
            self.assertEqual(output.getvalue().strip(), "[ERROR] Camera unavailable")


if __name__ == "__main__":
    unittest.main()
