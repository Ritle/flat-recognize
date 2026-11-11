import copy
import unittest

from scripts.compare_training_baseline import compare


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


if __name__ == "__main__":
    unittest.main()
