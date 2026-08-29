import unittest
from collections import Counter, defaultdict

import numpy as np

from source.multimodal import build_prompts, make_workspace_motion_panel
from source.preprocess import construct
from source.task_logic import build_multi_arm_role_assignment


class Scene3LogicTest(unittest.TestCase):
    def setUp(self):
        self.task = construct({"scenario": "scene3_dyad_causal_identification"})

    def test_legacy_names_resolve_to_dyad_scene(self):
        for legacy_name in (
            "scene3_triad_causal_identification",
            "scene3_triad_delay_invert",
        ):
            task = construct({"scenario": legacy_name})
            self.assertEqual(task["name"], "scene3_dyad_causal_identification")
            self.assertEqual(task["requested_name"], legacy_name)

    def test_scene_delay_matches_each_difficulty(self):
        for level, expected_delay in ((1, 3), (2, 2), (3, 1)):
            task = construct(
                {"scenario": "scene3_dyad_causal_identification", "level": level}
            )
            self.assertEqual(task["num_arms"], 2)
            self.assertEqual(len(task["distractors"]), 1)
            self.assertEqual(task["distractors"][0]["behavior"], "delay")
            self.assertEqual(task["distractors"][0]["delay"], expected_delay)

    def test_judgment_form_samples_direct_and_delayed_single_arms(self):
        for level, expected_delay in ((1, 3), (2, 2), (3, 1)):
            task = construct(
                {
                    "scenario": "scene3_dyad_causal_identification",
                    "test_type": "judgment",
                    "level": level,
                }
            )
            self.assertEqual(task["task_mode"], "single_binary")
            self.assertEqual(task["num_arms"], 1)
            options = task["visible_arm_behavior_options"]
            self.assertEqual(options[0]["behavior"]["behavior"], "direct")
            self.assertEqual(options[1]["behavior"]["behavior"], "delay")
            self.assertEqual(options[1]["behavior"]["delay"], expected_delay)
            self.assertFalse(task["annotate_candidates"])

    def test_two_consecutive_seeds_balance_self_and_delay_positions(self):
        target_positions = []
        behavior_positions = defaultdict(list)

        for seed in range(2):
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

        self.assertEqual(Counter(target_positions), Counter({1: 1, 2: 1}))
        self.assertEqual(Counter(behavior_positions["direct"]), Counter({1: 1, 2: 1}))
        self.assertEqual(Counter(behavior_positions["delay"]), Counter({1: 1, 2: 1}))
        self.assertNotIn("invert", behavior_positions)

    def test_target_override_places_delay_in_other_position(self):
        delay_positions = []

        for seed in range(4):
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

        self.assertEqual(Counter(delay_positions), Counter({1: 4}))

    def test_scene_judges_once_after_complete_diagnostic_sequence(self):
        self.assertEqual(self.task["episode_steps"], 8)
        self.assertEqual(self.task["judge_start_step"], 8)
        self.assertEqual(self.task["judge_interval_steps"], 8)
        self.assertEqual(
            self.task["visual_history_mode"],
            "workspace_motion_panels",
        )
        self.assertTrue(self.task["annotate_candidates"])

    def test_prompt_does_not_disclose_delay(self):
        prefix, suffix = build_prompts(
            ["candidate 1", "candidate 2"]
        )
        prompt = prefix + suffix
        self.assertNotIn("delay", prompt.lower())
        self.assertNotIn("previous step", prompt.lower())
        self.assertNotIn("difficulty", prompt.lower())

    def test_workspace_motion_panel_contains_before_after_and_signed_change(self):
        previous = np.full((96, 96, 3), 30, dtype=np.uint8)
        current = previous.copy()
        current[20:42, 8:24, :] = 220
        current[48:68, 70:88, :] = 6

        panel = make_workspace_motion_panel(
            previous,
            current,
            num_candidates=2,
            annotate=True,
        )

        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreater(panel.shape[0], 0)
        self.assertGreater(panel.shape[1], 0)
        self.assertTrue(np.any(np.all(panel == np.array([255, 132, 36]), axis=2)))
        self.assertTrue(np.any(np.all(panel == np.array([52, 128, 255]), axis=2)))
