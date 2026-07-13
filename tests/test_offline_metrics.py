import unittest

from source.offline_metrics import summarize_result_group


class OfflineMetricsTest(unittest.TestCase):
    def test_binary_metrics_separate_self_and_nonself(self):
        rows = [
            self._binary_row("a", True, True, 1),
            self._binary_row("b", True, False, 2),
            self._binary_row("c", False, True, 2),
            self._binary_row("d", False, True, 2),
        ]
        summary = summarize_result_group(rows, bootstrap_samples=100, seed=0)
        self.assertEqual(summary["accuracy"], 0.75)
        self.assertEqual(summary["self_recall"], 0.5)
        self.assertEqual(summary["nonself_recall"], 1.0)
        self.assertEqual(summary["balanced_accuracy"], 0.75)
        self.assertEqual(summary["self_attribution_rate"], 0.25)

    def test_multi_arm_metrics_report_position_bias(self):
        rows = [
            self._multi_row("a", 1, 1),
            self._multi_row("b", 1, 1),
            self._multi_row("c", 2, 1),
            self._multi_row("d", 2, 2),
            self._multi_row("e", 3, 1),
            self._multi_row("f", 3, 1),
        ]
        summary = summarize_result_group(rows, bootstrap_samples=100, seed=0)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["target_position_recall"]["1"], 1.0)
        self.assertEqual(summary["target_position_recall"]["2"], 0.5)
        self.assertEqual(summary["target_position_recall"]["3"], 0.0)
        self.assertEqual(
            summary["predicted_position_distribution"]["1"],
            0.833333,
        )
        self.assertEqual(summary["macro_position_recall"], 0.5)

    @staticmethod
    def _binary_row(episode_id, target_present, correct, choice):
        return {
            "episode_id": episode_id,
            "target_present": target_present,
            "target_index": 1 if target_present else None,
            "correct": correct,
            "valid": True,
            "choice": choice,
            "answer_options": ["yes, self", "no, non-self"],
        }

    @staticmethod
    def _multi_row(episode_id, target_index, choice):
        return {
            "episode_id": episode_id,
            "target_present": True,
            "target_index": target_index,
            "correct": target_index == choice,
            "valid": True,
            "choice": choice,
            "answer_options": ["candidate 1", "candidate 2", "candidate 3"],
        }


if __name__ == "__main__":
    unittest.main()
