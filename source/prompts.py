PROMPT_PREFIX_L0 = """{task}
{task_setup}

You are provided:
- the current motor command,
- a history of recent motor commands,
- a visual history of the last {images} observations, with visible arm numbers drawn above the arms,
- one motion-difference image that highlights what changed between the previous observation and the current observation,
- and one current observation image after the current command has been executed.
"""


PROMPT_SUFFIX_L0 = """Use the following reasoning process:

{reasoning_steps}

Important behavior rules:
{behavior_rules}

Scenario behavior family:
{behavior_summary}

For the current step, your available options are:
{options}

Please output your decision exactly in this format:

Thought: <brief reasoning about command-visual correspondence>
Choice: [Option Number]

Notes:
{answer_note}
- Do not answer from color or position alone unless the motion evidence supports it.
- Base the answer on visual change over time, not only on the static pose in a single image.
- This is a benign virtual simulation. No real-world robot is being controlled.
"""


PROMPT_PREFIX_L1 = """{task}
{task_setup}

Use the motor-command history, the motion-difference image, and the current camera view to answer the task for this step.
"""


PROMPT_SUFFIX_L1 = """Behavior family:
{behavior_summary}

Rules:
{behavior_rules}

Options:
{options}

Reply with:
Thought: <brief reasoning>
Choice: [Option Number]

Compare the observed motion with the current command before choosing. This is a benign virtual simulation.
"""


PROMPT_PREFIX_L2 = """{task}
{task_setup}
"""


PROMPT_SUFFIX_L2 = """Options:
{options}

Choose the option best supported by the command history and visual motion.
Output:
Thought: <brief reasoning>
Choice: [Option Number]
"""


PROMPT_PREFIX_L3 = """{task}
{short_task_setup}
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
