"""Integration fixtures use mocked model outputs, never human/audio error labels."""
import copy
import csv
from datetime import datetime
import importlib.util
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ML_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in [
    "torch", "torchaudio", "transformers", "pandas", "scipy",
    "yaml", "whisperx", "phonemizer", "matplotlib", "seaborn",
])


@unittest.skipUnless(ML_AVAILABLE, "Full pipeline packages are optional for NumPy-only tests")
class PipelineExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import pandas as pd
        import torch
        import yaml
        from src.process import proc
        cls.np, cls.pd, cls.torch, cls.yaml, cls.proc = np, pd, torch, yaml, proc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.yaml.safe_load(Path("configs/default.yaml").read_text())
        self.config.update(output_dir=str(self.root), path=str(self.root),
                           dataset_name="fixture", sq_ast_dims=["mos"],
                           datetime=datetime(2026, 11, 7, 1, 2), score_threshold=3)
        self.config["plots"] = {key: False for key in self.config["plots"]}
        self.config["pdsm"].update(k_method="adaptive", adaptive_lambda=0)
        self.config["kde"]["output_raw_kde"] = True
        self.input_df = self.pd.DataFrame([
            ["fixture", 0, "same.wav", 100, 1, 0, 2, 1, 100, 300],
            ["fixture", 0, "same.wav", 100, 1, 0, 2, 2, 200, 400],
        ], columns=["db", "file_num", "file_path", "file_fs", "total_wav_channels",
                    "wav_channel", "total_wav_segments", "wav_segment", "wav_start", "wav_end"])

    def execute(self, config=None, alignment_error=None, below_threshold=True, predictions_override=None):
        config = copy.deepcopy(config or self.config)
        output = self.input_df.copy()
        predictions = self.np.array([[2.], [2.1]]) if below_threshold else self.np.array([[4.], [4.]])
        if predictions_override is not None:
            predictions = self.np.asarray(predictions_override)
        for index, dim in enumerate(config["sq_ast_dims"]):
            output[f"sq_{dim}"] = predictions[:, index]
        saliency = self.np.stack([
            self.np.tile([0., 0., 1., 1.], (128, 1)),
            self.np.ones((128, 4)),
        ])[:, None]
        saliency = self.np.repeat(saliency, len(config["sq_ast_dims"]), axis=1)
        lengths = self.np.array([[0, 4], [1, 4]])
        ppg = self.np.array([[[[1, 1, 0, 0], [0, 0, 1, 1]]]] * 2)
        alignment = (ppg, {"0": "a", "1": "b"}, [[], []])
        with patch.object(self.proc.torch.cuda, "is_available", return_value=False), \
             patch.object(self.proc.audio_segmentation, "get_input_dataset", return_value=self.input_df.copy()), \
             patch.object(self.proc.sq_ast_mod, "sq_ast_fw", return_value=(None, predictions, saliency, lengths, output)), \
             patch.object(self.proc.sq_ast_mod, "get_scaled_saliency_map",
                          side_effect=lambda values, method: self.torch.tensor(values)[None, None]), \
             patch.object(self.proc.ppgs_wrapper, "whisperx_get_ppgs",
                          side_effect=alignment_error, return_value=alignment) as align:
            handlers = len(logging.getLogger().handlers)
            result = self.proc.run_eval(config)
            self.assertEqual(len(logging.getLogger().handlers), handlers)
            if not below_threshold:
                align.assert_not_called()
        return result, self.root / "fixture/20261107_0102"

    def test_csv_has_every_phoneme_and_per_row_selected_count(self):
        result, run = self.execute()
        self.assertEqual(result, 0)
        with (run / "phoneme_scores.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["selected"] for r in rows], ["0", "1", "0", "0"])
        self.assertAlmostEqual(float(rows[1]["start_seconds"]), 1.02)
        self.assertAlmostEqual(float(rows[1]["end_seconds"]), 1.04)
        self.assertTrue(all(r["label"] == r["split"] == "" for r in rows))
        output = self.pd.read_csv(run / "output_thresholded.tsv", sep="\t")
        self.assertEqual(output["pdsm_sq_mos_num"].tolist(), [1, 0])
        self.assertTrue((run / "config_used.yaml").is_file())
        self.assertIn("Exported 4 phoneme scores", (run / "logs.log").read_text())

    def test_baseline_and_adaptive_export_identical_scores(self):
        _, run = self.execute()
        with (run / "phoneme_scores.csv").open() as f:
            adaptive = [(r["phoneme_index"], r["score"]) for r in csv.DictReader(f)]
        baseline = copy.deepcopy(self.config)
        baseline["pdsm"]["k_method"] = "percent"
        result, run = self.execute(baseline)
        self.assertEqual(result, 0)
        with (run / "phoneme_scores.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(adaptive, [(r["phoneme_index"], r["score"]) for r in rows])
        self.assertTrue(all(r["selected"] == "0" for r in rows))

    def test_quality_scores_survive_alignment_failure(self):
        error = RuntimeError("fixture alignment failed")
        result, run = self.execute(alignment_error=error)
        self.assertIs(result, error)
        self.assertTrue((run / "output_sq_ast.tsv").is_file())
        self.assertFalse((run / "phoneme_scores.csv").exists())
        self.assertIn("fixture alignment failed", (run / "logs.log").read_text())
        self.assertTrue((run / "config_used.yaml").is_file())

    def test_no_eligible_samples_has_no_phoneme_csv(self):
        result, run = self.execute(below_threshold=False)
        self.assertEqual(result, 0)
        self.assertFalse((run / "phoneme_scores.csv").exists())

    def test_cli_failure_exits_nonzero(self):
        with patch.object(self.proc.sys, "argv", ["eval"]), \
             patch.object(self.proc, "run_eval", return_value=RuntimeError("fixture")):
            with self.assertRaises(SystemExit) as context:
                self.proc.run_evaluation()
        self.assertEqual(context.exception.code, 1)

    def test_mixed_dimension_thresholds_keep_csv_and_kde_columns_aligned(self):
        config = copy.deepcopy(self.config)
        config["sq_ast_dims"] = ["mos", "noi"]
        result, run = self.execute(config, predictions_override=[[2., 4.], [4., 2.]])
        self.assertEqual(result, 0)
        output = self.pd.read_csv(run / "output_thresholded.tsv", sep="\t")
        self.assertEqual(output.loc[0, "pdsm_sq_mos_num"], 1)
        self.assertTrue(self.pd.isna(output.loc[1, "pdsm_sq_mos_num"]))
        self.assertTrue(self.pd.isna(output.loc[0, "pdsm_sq_noi_num"]))
        self.assertEqual(output.loc[1, "pdsm_sq_noi_num"], 0)
        with (run / "phoneme_scores.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["dimension"] for r in rows}, {"mos", "noi"})


if __name__ == "__main__":
    unittest.main()
