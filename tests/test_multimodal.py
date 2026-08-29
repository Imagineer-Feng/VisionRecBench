import unittest

import numpy as np

from source.multimodal import (
    _fit_image,
    _workspace_crop,
    make_workspace_motion_panel,
)


class MultimodalEvidenceTest(unittest.TestCase):
    def test_fit_image_preserves_aspect_ratio_with_padding(self):
        source = np.full((80, 40, 3), [180, 30, 20], dtype=np.uint8)

        fitted = np.asarray(_fit_image(source, (60, 60)))
        foreground = np.any(fitted != np.array([12, 18, 24]), axis=2)
        rows, columns = np.where(foreground)

        self.assertEqual(rows.max() - rows.min() + 1, 60)
        self.assertEqual(columns.max() - columns.min() + 1, 30)

    def test_workspace_panel_has_three_compact_undistorted_columns(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220

        panel = make_workspace_motion_panel(
            previous,
            current,
            num_candidates=2,
            annotate=False,
        )

        self.assertEqual(panel.shape, (81, 144, 3))

    def test_judgment_and_choice_use_the_same_workspace_geometry(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220

        judgment = make_workspace_motion_panel(
            previous,
            current,
            num_candidates=1,
            annotate=False,
        )
        choice = make_workspace_motion_panel(
            previous,
            current,
            num_candidates=2,
            annotate=False,
        )

        self.assertEqual(judgment.shape, choice.shape)

    def test_workspace_crop_does_not_split_motion_at_image_midline(self):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[45:55, 30:66, 0] = 255

        cropped = _workspace_crop(image, "front")

        self.assertEqual(cropped.shape[1], image.shape[1])
        self.assertTrue(np.all(cropped[12:22, 30:66, 0] == 255))

    def test_production_panel_stays_on_three_api_tiles(self):
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)

        panel = make_workspace_motion_panel(
            image,
            image,
            num_candidates=2,
            camera_view="overhead",
        )

        self.assertEqual(panel.shape, (338, 1536, 3))


if __name__ == "__main__":
    unittest.main()
