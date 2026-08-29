import unittest

import numpy as np

from source.multimodal import _fit_image, make_candidate_motion_panel


class MultimodalEvidenceTest(unittest.TestCase):
    def test_fit_image_preserves_aspect_ratio_with_padding(self):
        source = np.full((80, 40, 3), [180, 30, 20], dtype=np.uint8)

        fitted = np.asarray(_fit_image(source, (60, 60)))
        foreground = np.any(fitted != np.array([12, 18, 24]), axis=2)
        rows, columns = np.where(foreground)

        self.assertEqual(rows.max() - rows.min() + 1, 60)
        self.assertEqual(columns.max() - columns.min() + 1, 30)

    def test_choice_panel_places_candidates_in_rows_without_distortion(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220

        panel = make_candidate_motion_panel(
            previous,
            current,
            num_candidates=2,
        )

        self.assertEqual(panel.shape, (192, 192, 3))

    def test_judgment_panel_uses_one_candidate_row(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220

        panel = make_candidate_motion_panel(
            previous,
            current,
            num_candidates=1,
        )

        self.assertEqual(panel.shape, (96, 192, 3))


if __name__ == "__main__":
    unittest.main()
