import unittest
from collections import Counter, defaultdict

import numpy as np

from scripts.inference import build_prompts, make_candidate_motion_panel
from source.preprocess import construct
from source.task_logic import build_multi_arm_role_assignment


class Scene3LogicTest(unittest.TestCase):
    def setUp(self):
        self.task = construct({"scenario": "scene3_triad_causal_identification"})

    def test_legacy_name_resolves_to_new_scene(self):
        task = construct({"scenario": "scene3_triad_delay_invert"})
        self.assertEqual(task["name"], "scene3_triad_causal_identification")
        self.assertEqual(task["requested_name"], "scene3_triad_delay_invert")

    def test_six_consecutive_seeds_balance_target_and_distractor_positions(self):
        target_positions = []
        behavior_positions = defaultdict(list)

        for seed in range(6):
            assignments = build_multi_arm_role_assignment(
                self.task["num_arms"],
                self.task["distractors"],
                seed=seed,
            )
            for item in assignments:
                behavior = item["behavior"]["behavior"]
                behavior_positions[behavior].append(item["index"])
                if item["role"] == "target":
                    target_positions.append(item["index"])

        self.assertEqual(Counter(target_positions), Counter({1: 2, 2: 2, 3: 2}))
        self.assertEqual(Counter(behavior_positions["direct"]), Counter({1: 2, 2: 2, 3: 2}))
        self.assertEqual(Counter(behavior_positions["delay"]), Counter({1: 2, 2: 2, 3: 2}))
        self.assertEqual(Counter(behavior_positions["invert"]), Counter({1: 2, 2: 2, 3: 2}))

    def test_target_override_keeps_target_fixed_but_still_permutates_distractors(self):
        delay_positions = []
        invert_positions = []

        for seed in range(6):
            assignments = build_multi_arm_role_assignment(
                self.task["num_arms"],
                self.task["distractors"],
                seed=seed,
                target_index=2,
            )
            self.assertEqual(assignments[1]["role"], "target")
            for item in assignments:
                behavior = item["behavior"]["behavior"]
                if behavior == "delay":
                    delay_positions.append(item["index"])
                elif behavior == "invert":
                    invert_positions.append(item["index"])

        self.assertEqual(Counter(delay_positions), Counter({1: 3, 3: 3}))
        self.assertEqual(Counter(invert_positions), Counter({1: 3, 3: 3}))

    def test_scene_judges_once_after_complete_diagnostic_sequence(self):
        self.assertEqual(self.task["episode_steps"], 8)
        self.assertEqual(self.task["judge_start_step"], 8)
        self.assertEqual(self.task["judge_interval_steps"], 8)
        self.assertEqual(self.task["visual_history_mode"], "candidate_motion_panels")
        self.assertTrue(self.task["annotate_candidates"])

    def test_prompt_describes_role_randomization_and_axis_signs(self):
        prefix, suffix = build_prompts(
            1,
            self.task,
            max_image_history=8,
        )
        prompt = prefix + suffix
        self.assertIn("candidate motion panel", prompt)
        self.assertIn("left/right position alone is not evidence", prompt)
        self.assertIn("joint-axis signs", prompt)
        self.assertNotIn("current motion-difference image", prompt)

    def test_candidate_motion_panel_contains_before_after_and_signed_change(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220
        current[48:68, 70:88, :] = 6

        panel = make_candidate_motion_panel(
            previous,
            current,
            num_candidates=3,
            annotate=True,
        )

        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreater(panel.shape[0], 0)
        self.assertGreater(panel.shape[1], 0)
        self.assertTrue(np.any(np.all(panel == np.array([255, 132, 36]), axis=2)))
        self.assertTrue(np.any(np.all(panel == np.array([52, 128, 255]), axis=2)))
