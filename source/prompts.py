PROMPT_PREFIX = """You are a simulated robotic agent performing visual self-recognition.

You are given a complete motor-command trace and time-ordered visual evidence of robotic-arm motion. Depending on the answer options, decide whether the visible arm is controlled by you or which candidate arm is controlled by you.

Use the entire sequence. Infer self-control from the consistency, predictability, and temporal relationship between motor commands and visible changes. A command label need not describe the observed physical joint literally, so assess reproducible command-to-motion relationships rather than relying on names alone.

Do not decide from option number, left/right position, color, total motion amount, a single step, or the final static pose. Treat every option as equally likely before examining the evidence. Visual evidence may be shown as motion-change views or as labeled before/after/change panels; use any labels and legends embedded in the images.
"""


PROMPT_SUFFIX = """Available options:
{options}

Choose the single option best supported by the complete command and visual evidence. Do not infer the answer from the wording or ordering of the options.

Reply exactly in this format, with the choice on the first line before any explanation:
Choice: [Option Number]
Thought: <brief evidence-based reasoning>
"""
