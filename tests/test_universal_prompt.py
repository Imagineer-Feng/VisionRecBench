import unittest

from source.multimodal import build_prompts
from source.preprocess import construct


class UniversalPromptTest(unittest.TestCase):
    def test_every_scene_and_level_uses_identical_prompt_text(self):
        scenarios = (
            "scene1_single_command_causality",
            "scene2_single_scrambled_stability",
            "scene3_dyad_causal_identification",
        )
        shared_options = ["first answer", "second answer"]
        expected = build_prompts(shared_options)
        for level in (1, 2, 3):
            for scenario in scenarios:
                construct({"scenario": scenario, "level": level})
                self.assertEqual(build_prompts(shared_options), expected)

    def test_prompt_does_not_reveal_difficulty_manipulations(self):
        prompt = "\n".join(build_prompts(["first answer", "second answer"])).lower()
        forbidden = (
            "easy",
            "medium",
            "hard",
            "difficulty",
            "mismatch",
            "delay",
            "previous step",
            "four cycles",
            "six steps",
            "mapping changes",
            "correct option",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
