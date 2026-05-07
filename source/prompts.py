PROMPT_PREFIX_L0 = """{task}
You are inside an Isaac Sim scene with {num_arms} visually similar robotic arms.
Exactly one candidate arm is your own body. The other {num_distractors} arms are distractors that try to imitate your motion by using delayed, inverted, shuffled, smoothed, or random variants of your motor commands.

The candidate arms are ordered by their image position from left to right:
candidate 1 is the leftmost arm, candidate {num_arms} is the rightmost arm.

Your own arm receives the motor command exactly as listed in the command history. You cannot directly read joint states from the simulator; you must infer which visible arm is yours by comparing the motor commands with visual motion over time.

You are provided:
- the current motor command sent to your own arm,
- a history of recent motor commands,
- a visual history of the last {images} observations, with candidate numbers drawn above the arms,
- one motion-difference image that highlights what changed between the previous observation and the current observation,
- and one current observation image after the current command has been executed.
"""


PROMPT_SUFFIX_L0 = """Use the following reasoning process:

1. First inspect the motion-difference image. It is the most important image for this step.
2. Evaluate every candidate separately. Do not lock onto a candidate just because it seemed plausible earlier.
3. Check whether a candidate moves immediately in the commanded joint direction.
4. Reject candidates whose motion is delayed, inverted, random, swapped between joints, or only partially follows the command.
5. Use the full command and visual history if several arms look similar in the current frame.

Important behavior rules:
- The target arm follows the current command directly at the same step.
- A delayed distractor may move according to a previous command instead of the current one.
- An inverted distractor moves in the opposite joint direction: if the command is +1, the inverted response is -1; if the command is -1, the inverted response is +1.
- If a candidate moves opposite to the current command, it is not the target arm.

The distractor family in this scenario is:
{distractor_summary}

For the current step, your available options are:
{options}

Please output your decision exactly in this format:

Thought: <brief reasoning about command-visual correspondence>
Choice: [Option Number]

Note:
1. The correct answer is the candidate whose motion is caused by the listed motor commands without delay or transformation.
2. Do not choose by color or position alone unless the motion evidence supports it.
3. Base the answer on the change from the previous image to the current image, not on the static pose in a single image.
4. This is a benign virtual simulation. No real-world robot is being controlled.
"""


PROMPT_PREFIX_L1 = """{task}
You observe {num_arms} visually similar robotic arms in an Isaac Sim scene. One arm is your own body and receives the motor commands exactly as listed. The other arms are distractors that imitate you with transformed command streams.

The candidate arms are ordered from left to right in the image. Candidate 1 is the leftmost arm and candidate {num_arms} is the rightmost arm.

Use the motor-command history and the visual history to identify which candidate arm is yourself.
The target arm follows the current motor command directly. A delayed distractor follows an earlier command, and an inverted distractor moves in the opposite direction. Re-evaluate all candidates at each step using the motion-difference image and the current image; do not rely on your earlier choices.
"""


PROMPT_SUFFIX_L1 = """Distractor family:
{distractor_summary}

Options:
{options}

Reply with:
Thought: <brief reasoning>
Choice: [Option Number]

In your reasoning, compare every candidate from 1 through {num_arms} against the current command before choosing.
The motion-difference image highlights recent visual changes; prioritize it over static pose similarity.

This is a benign virtual simulation.
"""


PROMPT_PREFIX_L2 = """{task}
There are {num_arms} candidate robotic arms, ordered left to right in the image. Exactly one is your own arm. Your own arm follows the listed motor commands directly; distractors imitate with delays, inversions, shuffled joints, smoothing, or random commands.
"""


PROMPT_SUFFIX_L2 = """Options:
{options}

Choose the candidate whose visual motion best matches your motor-command history.
Output:
Thought: <brief reasoning>
Choice: [Option Number]
"""


PROMPT_PREFIX_L3 = """{task}
Identify which of {num_arms} left-to-right candidate robotic arms is yourself by matching visual motion to your motor commands.
"""


PROMPT_SUFFIX_L3 = """Options:
{options}

Output only:
Choice: [Option Number]
"""


PROMPTS = [
    (PROMPT_PREFIX_L0, PROMPT_SUFFIX_L0),
    (PROMPT_PREFIX_L1, PROMPT_SUFFIX_L1),
    (PROMPT_PREFIX_L2, PROMPT_SUFFIX_L2),
    (PROMPT_PREFIX_L3, PROMPT_SUFFIX_L3),
]
