from dataclasses import dataclass


@dataclass
class Identification:
    text: str = ""
    choice: int = -1  # -1: cannot parse; -2: out of range
    valid: bool = False


def candidate_options(num_arms):
    return [""] + [f"candidate arm {i} from left to right" for i in range(1, num_arms + 1)]


def options_string(num_arms=None, labels=None):
    if labels is not None:
        return "\n".join(
            f"{i}. {label}"
            for i, label in enumerate(labels, start=1)
        )

    return "\n".join(
        f"{i}. candidate arm {i} from left to right"
        for i in range(1, num_arms + 1)
    )
