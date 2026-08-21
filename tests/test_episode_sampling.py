import unittest
from collections import Counter

import numpy as np

from source.episode_sampling import (
    STANDARD_SCENARIOS,
    build_episode_task,
)
from source.preprocess import construct
from source.task_logic import (
    build_multi_arm_role_assignment,
    configure_binary_answers,
    materialize_mapping_behavior,
    select_behavior_option,
)


class EpisodeSamplingTest(unittest.TestCase):
    def test_sampling_is_deterministic(self):
        first = build_episode_task(STANDARD_SCENARIOS[0], 7, base_seed=19)
        second = build_episode_task(STANDARD_SCENARIOS[0], 7, base_seed=19)
        self.assertEqual(first, second)

    def test_robust_split_has_unique_episode_signatures(self):
        for scenario in STANDARD_SCENARIOS:
            tasks = [
                build_episode_task(scenario, index, base_seed=0)
                for index in range(48)
            ]
            signatures = {task["episode_signature"] for task in tasks}
            self.assertEqual(len(signatures), 48)

    def test_sampling_changes_commands_pose_camera_and_lighting(self):
        scenario = "scene3_dyad_causal_identification"
        base = construct({"scenario": scenario})
        sampled = build_episode_task(scenario, 0, base_seed=0)
        self.assertNotEqual(
            sampled["command_sequence"],
            base["command_sequence"],
        )
        self.assertNotEqual(
            sampled["arm"]["initial_joint_positions"],
            base["arm"]["initial_joint_positions"],
        )
        self.assertNotEqual(sampled["camera_eye"], base["camera_eye"])
        self.assertNotEqual(
            sampled["key_light_intensity"],
            base["key_light_intensity"],
        )

    def test_sampled_initial_poses_stay_inside_joint_limits(self):
        for scenario in STANDARD_SCENARIOS:
            for index in range(24):
                task = build_episode_task(scenario, index, base_seed=4)
                initial = np.asarray(
                    task["arm"]["initial_joint_positions"],
                    dtype=float,
                )
                limits = np.asarray(task["arm"]["joint_limits"], dtype=float)
                self.assertTrue(np.all(initial >= limits[:, 0]))
                self.assertTrue(np.all(initial <= limits[:, 1]))

    def test_binary_conditions_and_answer_positions_are_balanced(self):
        for scenario in STANDARD_SCENARIOS[:2]:
            combinations = Counter()
            mapping_options = Counter()
            for index in range(48):
                task = build_episode_task(scenario, index, base_seed=0)
                options = task["visible_arm_behavior_options"]
                selected_index = select_behavior_option(
                    options,
                    seed=task["seed"],
                    strategy=task["behavior_selection"],
                )
                selected = options[selected_index]
                _, answer_index = configure_binary_answers(
                    task["answer_options"],
                    selected["target_present"],
                    task["seed"],
                    shuffle=True,
                )
                combinations[(selected["target_present"], answer_index)] += 1
                behavior = materialize_mapping_behavior(
                    selected["behavior"],
                    seed=task["seed"],
                    behavior_option_count=len(options),
                )
                if "sampled_mapping_option" in behavior:
                    mapping_options[
                        (selected["target_present"], behavior["sampled_mapping_option"])
                    ] += 1

            self.assertEqual(set(combinations.values()), {12})
            if mapping_options:
                self.assertEqual(set(mapping_options.values()), {8})

    def test_scene3_target_and_distractor_roles_are_balanced(self):
        positions = Counter()
        behavior_positions = Counter()
        for index in range(48):
            task = build_episode_task(
                "scene3_dyad_causal_identification",
                index,
                base_seed=0,
            )
            assignments = build_multi_arm_role_assignment(
                task["num_arms"],
                task["distractors"],
                task["seed"],
            )
            for assignment in assignments:
                behavior = assignment["behavior"]["behavior"]
                behavior_positions[(behavior, assignment["index"])] += 1
                if assignment["role"] == "target":
                    positions[assignment["index"]] += 1

        self.assertEqual(positions, Counter({1: 24, 2: 24}))
        self.assertEqual(
            behavior_positions,
            Counter(
                {
                    ("direct", 1): 24,
                    ("direct", 2): 24,
                    ("delay", 1): 24,
                    ("delay", 2): 24,
                }
            ),
        )

    def test_scene2_preserves_one_repeatable_four_command_cycle(self):
        task = build_episode_task(
            "scene2_single_scrambled_stability",
            11,
            base_seed=0,
        )
        self.assertEqual(len(task["command_sequence"]), 4)
        supports = []
        for command in task["command_sequence"]:
            nonzero = np.flatnonzero(np.asarray(command["delta"]))
            self.assertEqual(len(nonzero), 1)
            supports.append(int(nonzero[0]))
        self.assertCountEqual(supports, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
