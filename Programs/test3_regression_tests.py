import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("test3.py")
SPEC = importlib.util.spec_from_file_location("test3_module", MODULE_PATH)
TEST3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TEST3)


class TestLmlExtraction(unittest.TestCase):
    def test_extract_lml_scalars_from_2d_float_history(self):
        raw = np.array([
            [np.nan, np.nan, 162.1],
            [np.nan, 161.9, np.nan],
        ], dtype=np.float64)
        extracted = TEST3._extract_lml_scalars(raw)
        np.testing.assert_allclose(extracted, [162.1, 161.9])

    def test_extract_lml_scalars_from_object_history(self):
        raw = np.array([
            [list([np.nan, np.nan, 117.2]), list([np.nan, 117.3])],
        ], dtype=object)
        extracted = TEST3._extract_lml_scalars(raw)
        np.testing.assert_allclose(extracted, [117.2, 117.3])


class TestPeakMatching(unittest.TestCase):
    def test_match_observed_and_predicted_peaks_penalizes_far_peak(self):
        matched_fwhms, matched_freqs, details = TEST3.match_observed_and_predicted_peaks(
            obs_freqs=[0.52],
            obs_fwhms=[0.075],
            pred_freqs=[0.78],
            pred_fwhms=[0.060],
            max_distance=0.12,
        )
        self.assertAlmostEqual(matched_freqs[0], 0.57)
        self.assertAlmostEqual(matched_fwhms[0], 0.15)
        self.assertEqual(details[0]["status"], "penalized")

    def test_summarize_peak_matching_uses_multi_peak_matching(self):
        summary = TEST3.summarize_peak_matching(
            obs_freqs=[0.523, 0.777],
            obs_fwhms=[0.075, 0.075],
            pred_freqs=[0.513, 0.752],
            pred_fwhms=[0.071, 0.071],
            max_distance=0.12,
        )
        self.assertEqual(summary["n_penalized"], 0)
        self.assertLess(summary["peak_mae_ghz"], 30.0)
        self.assertLess(summary["peak_max_err_ghz"], 30.0)


if __name__ == "__main__":
    unittest.main()
