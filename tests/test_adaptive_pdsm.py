import csv
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.compare_pdsm import compare, binary_metrics
from src.process.pdsm import PDSM, l2_norm, select_indices
from src.process.phoneme_export import PHONEME_SCORE_FIELDS, build_export_rows


def fixture_rows(file_path="dev.wav", split="", labels=None, n=20, dimension="mos"):
    rows = []
    for index in range(n):
        row = dict.fromkeys(PHONEME_SCORE_FIELDS, "")
        row.update(
            dataset="fixture", file_path=file_path, file_num=0, wav_channel=0,
            wav_segment=0, wav_start=0, wav_end=10000, file_fs=1000,
            dimension=dimension, phoneme_index=index, phoneme_count=n,
            phoneme_id=index, phoneme=f"p{index}", start_frame=index,
            end_frame=index, start_seconds=index * .01,
            end_seconds=(index + 1) * .01, score=index, selected=0,
            label="" if labels is None else labels[index],
            split="" if labels is None else split,
        )
        rows.append(row)
    return rows


class SelectionTests(unittest.TestCase):
    def test_adaptive_population_std_and_strict_cutoff(self):
        # mean=1, population std=1, cutoff=2. Equality is not selected.
        chosen, cutoff = select_indices([0, 2], "adaptive", adaptive_lambda=1)
        self.assertEqual(cutoff, 2)
        self.assertEqual(chosen.tolist(), [])

    def test_adaptive_selects_high_outlier(self):
        chosen, cutoff = select_indices([1, 2, 3, 10], "adaptive", adaptive_lambda=1)
        self.assertEqual(chosen.tolist(), [3])
        self.assertAlmostEqual(cutoff, 4 + np.sqrt(12.5))

    def test_adaptive_lambda_changes_selection(self):
        self.assertEqual(select_indices([0, 1, 2, 3], "adaptive", adaptive_lambda=0)[0].tolist(), [3, 2])
        self.assertEqual(select_indices([0, 1, 2, 3], "adaptive", adaptive_lambda=2)[0].tolist(), [])

    def test_equal_singleton_and_empty_select_nothing(self):
        for scores in [[], [5], [5, 5, 5], [0, 0]]:
            with self.subTest(scores=scores):
                chosen, _ = select_indices(scores, "adaptive")
                self.assertEqual(chosen.tolist(), [])

    def test_adaptive_ties_keep_time_order(self):
        chosen, _ = select_indices([0, 5, 5, 0], "adaptive", adaptive_lambda=0)
        self.assertEqual(chosen.tolist(), [1, 2])

    def test_percent_floor_zero_does_not_select_everything(self):
        for n in [0, 1, 3, 9]:
            self.assertEqual(select_indices(np.arange(n), "percent", .1)[0].tolist(), [])

    def test_percent_floor_and_time_ties(self):
        self.assertEqual(select_indices(np.arange(19), "percent", .1)[0].tolist(), [18])
        self.assertEqual(select_indices([1] * 20, "percent", .1)[0].tolist(), [0, 1])

    def test_count_zero_and_large_count(self):
        self.assertEqual(select_indices([2, 1], "threshold", 0)[0].tolist(), [])
        self.assertEqual(select_indices([2, 1], "threshold", 10)[0].tolist(), [0, 1])

    def test_invalid_rules_and_parameters(self):
        cases = [
            ("unknown", .1, 1), ("percent", -.1, 1), ("percent", 1.1, 1),
            ("threshold", 1.5, 1), ("threshold", -1, 1),
            ("adaptive", .1, -1), ("adaptive", .1, float("nan")),
            ("adaptive", .1, float("inf")), ("percent", True, 1),
        ]
        for method, k, scale in cases:
            with self.subTest(method=method, k=k, scale=scale):
                with self.assertRaises(ValueError):
                    select_indices([1, 2], method, k, scale)

    def test_nonfinite_or_nonvector_scores_rejected(self):
        for scores in [[float("nan")], [float("inf")], [[1, 2]]]:
            with self.assertRaises(ValueError):
                select_indices(scores, "adaptive")


class PoolingAndExportTests(unittest.TestCase):
    def test_single_frame_phoneme_is_scored_and_painted(self):
        saliency = np.array([[2., 5., 1.]])
        ppg = np.eye(3)
        mask, selected, records = PDSM(
            saliency, ppg, {"0": "a", "1": "b", "2": "c"},
            np.abs, np.sum, "threshold", 1, return_scores=True)
        self.assertEqual(selected, [(1, "b", 1, 1)])
        self.assertEqual([r["score"] for r in records], [2, 5, 1])
        self.assertEqual(mask.tolist(), [[0, .5, 0]])

    def test_inclusive_final_frame_contributes(self):
        _, _, records = PDSM(
            np.array([[1., 2., 3., 4.]]),
            np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
            {"0": "a", "1": "b"}, np.abs, np.sum, "percent", 1,
            return_scores=True)
        self.assertEqual([r["score"] for r in records], [3, 7])
        self.assertEqual([(r["start_frame"], r["end_frame"]) for r in records], [(0, 1), (2, 3)])

    def test_default_tuple_api_is_preserved(self):
        result = PDSM(np.ones((2, 1)), np.ones((1, 1)), {"0": "a"},
                      np.abs, l2_norm, "percent", .1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], [])
        self.assertTrue((result[0] == 0).all())

    def test_all_equal_adaptive_keeps_empty_mask_and_all_scores(self):
        mask, selected, records = PDSM(
            np.ones((1, 4)), np.eye(4), {str(i): str(i) for i in range(4)},
            np.abs, np.sum, "adaptive", .1, return_scores=True)
        self.assertEqual(selected, [])
        self.assertEqual(len(records), 4)
        self.assertTrue((mask == 0).all())
        self.assertFalse(any(r["selected"] for r in records))

    def test_empty_and_invalid_time_axes(self):
        result = PDSM(np.empty((2, 0)), np.empty((3, 0)), {}, np.abs, np.sum,
                      "adaptive", .1, return_scores=True)
        self.assertEqual(result[1:], ([], []))
        with self.assertRaises(ValueError):
            PDSM(np.zeros((2, 3)), np.zeros((3, 2)), {}, np.abs, np.sum, "adaptive", .1)

    def test_export_seconds_include_last_frame_and_original_offset(self):
        row = {"file_path": "x.wav", "file_num": 0, "wav_channel": 1, "wav_segment": 2,
               "file_fs": 1000, "wav_start": 2000, "wav_end": 2040, "sq_mos": 2.3}
        config = {"dataset_name": "demo", "pdsm": {"k_method": "adaptive", "k": .1,
                  "adaptive_lambda": 1, "preprocess": "abs", "pool": "sum"},
                  "saliency": {"saliency_method": "GradCAM"}}
        records = [{"start_frame": 2, "end_frame": 3, "score": 4., "selected": 1}]
        exported = build_export_rows(row, "mos", records, config)
        self.assertAlmostEqual(exported[0]["start_seconds"], 2.02)
        self.assertAlmostEqual(exported[0]["end_seconds"], 2.04)
        self.assertEqual(exported[0]["label"], "")
        self.assertEqual(exported[0]["split"], "")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_compare(self, rows):
        path = self.root / "scores.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PHONEME_SCORE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return compare(path, self.root / "comparison")

    def test_unlabelled_data_has_counts_but_no_accuracy(self):
        result = self.run_compare(fixture_rows())
        self.assertEqual(len(result["comparison"]), 12)
        self.assertTrue(all(r["f1"] is None for r in result["comparison"]))
        self.assertEqual(result["selection"]["status"], "not_selected")
        self.assertIsNone(result["selection"]["best_adaptive"])
        top10 = next(r for r in result["comparison"] if r["method"] == "top_10pct" and r["dimension"] == "ALL")
        self.assertEqual(top10["n_selected"], 2)
        with (self.root / "comparison/comparison.csv").open() as f:
            self.assertTrue(all(r["precision"] == "" for r in csv.DictReader(f)))
        with (self.root / "comparison/selections.csv").open() as f:
            selections = list(csv.DictReader(f))
        self.assertEqual(len(selections), 120)
        self.assertTrue(all(r["label"] == "" for r in selections))
        self.assertEqual(result["n_files"], 1)

    def test_dev_selection_never_uses_test_and_tie_prefers_precision(self):
        dev = fixture_rows(labels=[0] * 18 + [1, 1], split="dev")
        test = fixture_rows("test.wav", labels=[1, 1] + [0] * 18, split="test")
        result = self.run_compare(dev + test)
        chosen = result["selection"]
        self.assertEqual(chosen["best_overall"]["method"], "top_10pct")
        self.assertEqual(chosen["best_percent"]["dev"]["f1"], 1)
        self.assertEqual(chosen["best_overall"]["test"]["f1"], 0)
        self.assertEqual(chosen["best_adaptive"]["method"], "adaptive_1.5")
        self.assertEqual(chosen["best_adaptive"]["dev"]["precision"], 1)
        saved = json.loads((self.root / "comparison/selected_on_dev.json").read_text())
        self.assertEqual(saved["best_overall"]["method"], "top_10pct")

    def test_no_positive_dev_does_not_select_method(self):
        result = self.run_compare(fixture_rows(labels=[0] * 20, split="dev"))
        self.assertEqual(result["selection"]["status"], "not_selected")
        self.assertIn("no positive", result["selection"]["reason"])

    def test_test_only_does_not_select_method(self):
        result = self.run_compare(fixture_rows(labels=[0] * 18 + [1, 1], split="test"))
        self.assertEqual(result["selection"]["status"], "not_selected")

    def test_all_is_micro_sum_across_dimensions(self):
        rows = fixture_rows(labels=[0] * 18 + [1, 1], split="dev")
        rows += fixture_rows(labels=[0] * 19 + [1], split="dev", dimension="noi")
        result = self.run_compare(rows)
        selected = [r for r in result["comparison"] if r["method"] == "top_10pct"]
        all_row = next(r for r in selected if r["dimension"] == "ALL")
        parts = [r for r in selected if r["dimension"] != "ALL"]
        self.assertEqual(all_row["n_phonemes"], 40)
        self.assertEqual(all_row["tp"], sum(r["tp"] for r in parts))
        self.assertEqual(all_row["fp"], sum(r["fp"] for r in parts))
        self.assertEqual(all_row["n_files"], 1)

    def test_time_ties_are_independent_of_csv_row_order(self):
        rows = fixture_rows()
        for row in rows:
            row["score"] = 1
        self.run_compare(rows[::-1])
        with (self.root / "comparison/selections.csv").open() as f:
            selected = {int(r["phoneme_index"]) for r in csv.DictReader(f)
                        if r["method"] == "top_10pct" and r["method_selected"] == "1"}
        self.assertEqual(selected, {0, 1})

    def test_file_cannot_cross_dev_test_even_across_dimensions(self):
        rows = fixture_rows(labels=[0] * 20, split="dev")
        rows += fixture_rows(labels=[0] * 20, split="test", dimension="noi")
        with self.assertRaisesRegex(ValueError, "both dev and test"):
            self.run_compare(rows)
        self.assertFalse((self.root / "comparison").exists())

    def test_partially_labelled_group_rejected(self):
        rows = fixture_rows()
        rows[0].update(label=1, split="dev")
        with self.assertRaisesRegex(ValueError, "fully labelled"):
            self.run_compare(rows)

    def test_missing_label_or_split_rejected(self):
        for update in [{"label": 1}, {"split": "dev"}]:
            rows = fixture_rows()
            rows[0].update(update)
            with self.assertRaisesRegex(ValueError, "both"):
                self.run_compare(rows)

    def test_invalid_labels_rejected(self):
        rows = fixture_rows(labels=[0] * 20, split="dev")
        rows[0]["label"] = .5
        with self.assertRaises(ValueError):
            self.run_compare(rows)

    def test_incomplete_group_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete group"):
            self.run_compare(fixture_rows()[:-1])

    def test_duplicate_phoneme_rejected(self):
        rows = fixture_rows()
        with self.assertRaisesRegex(ValueError, "indices"):
            self.run_compare(rows + [rows[-1]])

    def test_nonfinite_score_rejected(self):
        rows = fixture_rows()
        rows[0]["score"] = "nan"
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_compare(rows)

    def test_empty_csv_rejected(self):
        with self.assertRaisesRegex(ValueError, "no phoneme"):
            self.run_compare([])

    def test_confusion_matrix_and_zero_denominators(self):
        metrics = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual([metrics[k] for k in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(metrics["precision"], .5)
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["f1"], .5)
        self.assertEqual(metrics["fpr"], .5)
        no_selection = binary_metrics([1, 1], [0, 0])
        self.assertEqual(no_selection["f1"], 0)
        self.assertIsNone(no_selection["fpr"])
        negative_only = binary_metrics([0, 0], [0, 0])
        self.assertEqual(negative_only["recall"], 0)
        self.assertEqual(negative_only["fpr"], 0)

    def test_fractional_binary_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([.5], [1])
        with self.assertRaises(ValueError):
            binary_metrics([1], [2])

    def test_comparison_import_needs_no_ml_stack(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import src.compare_pdsm,sys; print(any(m in sys.modules for m in ['torch','pandas','whisperx']))",
        ], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_normalized_file_alias_cannot_leak(self):
        rows = fixture_rows(labels=[0] * 20, split="dev")
        rows += fixture_rows("./dev.wav", labels=[0] * 20, split="test")
        with self.assertRaisesRegex(ValueError, "both dev and test"):
            self.run_compare(rows)

    def test_phoneme_whitespace_and_raw_scores_are_preserved(self):
        rows = fixture_rows()
        rows[0]["phoneme"] = " "
        rows[0]["score"] = " 0.0 "
        self.run_compare(rows)
        with (self.root / "comparison/selections.csv").open() as f:
            exported = [r for r in csv.DictReader(f) if r["phoneme_index"] == "0"]
        self.assertEqual(len(exported), 6)
        self.assertTrue(all(r["phoneme"] == " " and r["score"] == " 0.0 " for r in exported))


if __name__ == "__main__":
    unittest.main()
