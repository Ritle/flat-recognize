import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.compare_training_baseline import compare, compare_with_clean_control, main


class CompareTrainingBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rules = dict(max_base_mean_iou_drop=0.005, max_base_structural_class_iou_drop=0.005,
                          new_wall_must_not_drop=True, new_door_and_window_must_improve=True)
        self.before = {}
        for source, count in (("svg", 400), ("raster", 2)):
            iou = dict(floor=0.99, wall=0.98, door=0.96, window=0.98) if source == "svg" else dict(
                floor=0.9, wall=0.15, door=0.0, window=0.0)
            self.before[source] = dict(split="val", source=source, sample_count=count,
                                       limit_per_source=None, sources=[dict(source=source, samples=["same-hashes"])],
                                       evaluated_pixels=100, ignored_pixels=0, device="cpu", iou=iou,
                                       miou=sum(iou.values()) / 4, checkpoint_sha256="original",
                                       config=dict(data=dict(image_size=[512, 512], normalize=True, letterbox=True),
                                                   model=dict(encoder_name="resnet34"), optim=dict(batch_size=2)))
        self.after = copy.deepcopy(self.before)
        for row in self.after.values():
            row["checkpoint_sha256"] = "candidate"
        self.after["raster"]["iou"].update(wall=0.3, door=0.1, window=0.2)
        self.refresh()

    def refresh(self):
        for row in self.after.values():
            row["miou"] = sum(row["iou"].values()) / 4

    def test_improved_candidate_only_becomes_eligible_for_pipeline_review(self):
        result = compare(self.before, self.after, self.rules)
        self.assertTrue(result["eligible_for_pipeline_check"])
        self.assertFalse(result["deploy"])
        self.assertFalse(result["test_used"])

    def test_background_gain_cannot_hide_regressed_openings(self):
        self.after["svg"]["iou"].update(floor=1.0, door=0.95)
        self.after["raster"]["iou"]["window"] = 0
        self.refresh()
        result = compare(self.before, self.after, self.rules)
        self.assertFalse(result["eligible_for_pipeline_check"])
        self.assertIn("base_door_iou_regression", result["failed_checks"])
        self.assertIn("new_window_iou_did_not_improve", result["failed_checks"])

    def test_changed_cohort_preprocessing_or_partial_evaluation_cannot_be_compared(self):
        for change in ("cohort", "pixels", "resize", "svg_mode", "limited", "weights"):
            after = copy.deepcopy(self.after)
            if change == "cohort":
                after["raster"]["sources"][0]["samples"] = ["different-mask-hash"]
            elif change == "pixels":
                after["raster"]["ignored_pixels"] = 42
            elif change == "resize":
                after["svg"]["config"]["data"]["image_size"] = [384, 384]
            elif change == "limited":
                after["svg"]["limit_per_source"] = 2
            elif change == "svg_mode":
                after["svg"]["config"]["data"]["svg_input_mode"] = "full"
            else:
                after["raster"]["checkpoint_sha256"] = "other-candidate"
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(self.before, after, self.rules)

    def test_invalid_policy_cannot_silently_disable_regression_checks(self):
        for key, value in (("max_base_mean_iou_drop", float("nan")),
                           ("max_base_structural_class_iou_drop", -1),
                           ("new_wall_must_not_drop", "false")):
            rules = {**self.rules, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(self.before, self.after, rules)

    def full_scene_reports(self):
        before, after = copy.deepcopy(self.before), copy.deepcopy(self.after)
        clean_before, clean_after = copy.deepcopy(before["svg"]), copy.deepcopy(after["svg"])
        before["svg"]["config"]["data"]["svg_input_mode"] = "full"
        after["svg"]["config"]["data"]["svg_input_mode"] = "full"
        return before, after, clean_before, clean_after

    def test_full_scene_gains_and_preserved_clean_control_allow_only_pipeline_review(self):
        result = compare_with_clean_control(*self.full_scene_reports(), self.rules, 0.005)
        self.assertTrue(result["eligible_for_pipeline_check"])
        self.assertTrue(result["clean_control"]["eligible_for_pipeline_check"])
        self.assertFalse(result["deploy"])

    def test_full_scene_gain_cannot_hide_clean_opening_regression(self):
        before, after, clean_before, clean_after = self.full_scene_reports()
        clean_after["iou"].update(floor=1.0, window=0.96)
        clean_after["miou"] = sum(clean_after["iou"].values()) / 4
        result = compare_with_clean_control(before, after, clean_before, clean_after, self.rules, 0.005)
        self.assertFalse(result["eligible_for_pipeline_check"])
        self.assertIn("clean_control_base_window_iou_regression", result["failed_checks"])

    def test_borrowed_clean_score_from_original_weights_is_rejected(self):
        before, after, clean_before, clean_after = self.full_scene_reports()
        clean_after["checkpoint_sha256"] = "original"
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            compare_with_clean_control(before, after, clean_before, clean_after, self.rules, 0.005)

    def test_clean_control_cannot_be_replaced_with_full_scene_evaluation(self):
        before, after, clean_before, clean_after = self.full_scene_reports()
        clean_before["config"]["data"]["svg_input_mode"] = "full"
        clean_after["config"]["data"]["svg_input_mode"] = "full"
        with self.assertRaisesRegex(ValueError, "structural input reports"):
            compare_with_clean_control(before, after, clean_before, clean_after, self.rules, 0.005)

    def test_cli_cannot_save_full_scene_approval_without_clean_control(self):
        before, after, _, _ = self.full_scene_reports()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, reports in (("before", before), ("after", after)):
                (root / name).mkdir()
                for source, report in reports.items():
                    (root / name / f"val-{source}.json").write_text(json.dumps(report))
            protocol = {"initial_checkpoint_sha256": "original", "screening_rules": self.rules}
            path, output = root / "protocol.json", root / "result.json"
            argv = ["compare", "--baseline", str(root / "before"), "--candidate", str(root / "after"),
                    "--protocol", str(path), "--output", str(output)]
            for defined in (False, True):
                if defined:
                    protocol["clean_control_baseline"] = str(root / "recorded-baseline/summary.json")
                path.write_text(json.dumps(protocol))
                with self.subTest(defined=defined), patch("sys.argv", argv), self.assertRaises(ValueError):
                    main()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
