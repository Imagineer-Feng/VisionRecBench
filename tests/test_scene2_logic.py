import unittest
from collections import Counter

import numpy as np

from source.multimodal import build_prompts
from source.preprocess import construct
from source.task_logic import (
    apply_mapped_behavior,
    configure_binary_answers,
    materialize_mapping_behavior,
    select_behavior_option,
)


class Scene2LogicTest(unittest.TestCase):
    def setUp(self):
        self.task = construct({"scenario": "scene2_single_scrambled_stability"})
        self.behavior_options = self.task["visible_arm_behavior_options"]

    def test_legacy_name_resolves_to_new_scene(self):
        task = construct({"scenario": "scene2_single_scrambled_fixed"})
        self.assertEqual(task["name"], "scene2_single_scrambled_stability")
        self.assertEqual(task["requested_name"], "scene2_single_scrambled_fixed")

    def test_four_consecutive_seeds_balance_condition_and_answer_position(self):
        combinations = []
        for seed in range(4):
            option_index = select_behavior_option(
                self.behavior_options,
                seed=seed,
                strategy=self.task["behavior_selection"],
            )
            target_present = self.behavior_options[option_index]["target_present"]
            _, answer_index = configure_binary_answers(
                self.task["answer_options"],
                target_present=target_present,
                seed=seed,
                shuffle=self.task["shuffle_answer_options"],
            )
            combinations.append((target_present, answer_index))

        self.assertCountEqual(
            combinations,
            [(True, 1), (True, 2), (False, 1), (False, 2)],
        )

    def test_mapping_options_are_deranged_permutations(self):
        mappings = self.behavior_options[0]["behavior"]["mapping_options"]
        for mapping in mappings:
            matrix = np.asarray(mapping)
            np.testing.assert_array_equal(matrix.sum(axis=0), np.ones(4))
            np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(4))
            np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))

    def test_stable_mapping_repeats_same_response_across_four_cycles(self):
        behavior = materialize_mapping_behavior(
            self.behavior_options[0]["behavior"],
            seed=0,
            behavior_option_count=2,
        )
        command = np.array([1.0, 0.0, 0.0, 0.0])
        responses = [
            apply_mapped_behavior(behavior, command, step, command_dim=4)
            for step in (0, 4, 8, 12)
        ]
        np.testing.assert_array_equal(responses[0], responses[1])
        np.testing.assert_array_equal(responses[1], responses[2])

    def test_nonself_mapping_repetition_matches_each_difficulty(self):
        expected_patterns = {
            1: [1, 1, 1, 1],
            2: [2, 1, 1],
            3: [3, 1],
        }
        for level, expected_counts in expected_patterns.items():
            task = construct(
                {"scenario": "scene2_single_scrambled_stability", "level": level}
            )
            behavior = materialize_mapping_behavior(
                task["visible_arm_behavior_options"][1]["behavior"],
                seed=1,
                behavior_option_count=2,
            )
            mapping_counts = sorted(
                Counter(
                    tuple(np.asarray(mapping).ravel())
                    for mapping in behavior["mappings"]
                ).values(),
                reverse=True,
            )
            self.assertEqual(mapping_counts, expected_counts)

    def test_each_cycle_has_same_total_motion_budget(self):
        for option_index, option in enumerate(self.behavior_options):
            behavior = materialize_mapping_behavior(
                option["behavior"],
                seed=option_index,
                behavior_option_count=2,
            )
            cycle_totals = []
            for cycle_index in range(4):
                responses = [
                    apply_mapped_behavior(
                        behavior,
                        np.eye(4)[axis],
                        command_index=cycle_index * 4 + axis,
                        command_dim=4,
                    )
                    for axis in range(4)
                ]
                cycle_totals.append(np.sum(responses, axis=0))

            for total in cycle_totals:
                np.testing.assert_array_equal(total, np.ones(4))

    def test_prompt_does_not_disclose_mapping_conditions(self):
        prefix, suffix = build_prompts(
            list(reversed(self.task["answer_options"]))
        )
        prompt = prefix + suffix
        self.assertNotIn("non-self case", prompt.lower())
        self.assertNotIn("changes between cycles", prompt.lower())
        self.assertNotIn("difficulty", prompt.lower())
        self.assertNotIn("Option 1 means", prompt)

    def test_scene_contains_four_complete_four_axis_cycles(self):
        self.assertEqual(self.task["episode_steps"], 16)
        self.assertEqual(self.task["judge_start_step"], 16)
        self.assertEqual(self.task["judge_interval_steps"], 16)
        self.assertEqual(self.task["visual_history_mode"], "motion_diffs")
        self.assertFalse(self.task["annotate_candidates"])
        commands = np.asarray(
            [item["delta"] for item in self.task["command_sequence"]]
        )
        np.testing.assert_array_equal(commands, np.eye(4))

    def test_choice_form_compares_stable_and_switching_mappings(self):
        task = construct(
            {
                "scenario": "scene2_single_scrambled_stability",
                "test_type": "choice",
                "level": 3,
            }
        )
        self.assertEqual(task["task_mode"], "multi_arm")
        self.assertEqual(task["num_arms"], 2)
        self.assertEqual(task["target_behavior"]["behavior"], "mapped_direct")
        self.assertEqual(
            task["distractors"][0]["behavior"],
            "mapped_cycle_switch",
        )
        self.assertEqual(task["distractors"][0]["mapping_pattern"], [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
