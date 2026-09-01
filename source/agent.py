import base64
import io
import os
import random
import re
import time

import numpy as np
from PIL import Image

from source.action import Identification


SYSTEM_MESSAGE = (
    "You are evaluating a benign Isaac Sim scene. No real-world robot is being "
    "controlled. Compare the visible simulated robot motion with the provided "
    "motor-command trace and choose the requested answer option."
)
IMAGE_DETAIL = "high"
EVALUATION_PROTOCOL_BASE = "eval_v2"
DEFAULT_DECODING_PROFILE = "compatible"
DECODING_PROFILES = {
    # Common denominator for cross-provider OpenAI-compatible endpoints.
    "compatible": {
        "max_completion_tokens": 256,
    },
    # Optional sensitivity profile for models that explicitly support it.
    "temperature_zero": {
        "temperature": 0.0,
        "max_completion_tokens": 256,
    },
}
DECODING_PROFILE_IDS = tuple(DECODING_PROFILES)


def generation_parameters_for_profile(profile):
    try:
        return dict(DECODING_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"Unsupported decoding profile: {profile!r}") from exc


def evaluation_protocol_version(profile):
    generation_parameters_for_profile(profile)
    return f"{EVALUATION_PROTOCOL_BASE}_{profile}"


EVALUATION_PROTOCOL_VERSION = evaluation_protocol_version(
    DEFAULT_DECODING_PROFILE
)
OPENAI_GENERATION_PARAMETERS = generation_parameters_for_profile(
    DEFAULT_DECODING_PROFILE
)


def image_to_data_url(image):
    if isinstance(image, str):
        with open(image, "rb") as f:
            payload = base64.b64encode(f.read()).decode("utf-8")
    else:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def parse_choice(text):
    if not text:
        return -1

    normalized = re.sub(r"[*_`#]", "", text)
    patterns = [
        r"Choice\s*:?\s*\[?\s*(?:Option\s*)?(\d+)\s*\]?",
        r"candidate\s+arm\s+(\d+)",
        r"candidate\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return -1


def summarize_content(content_items):
    return " ".join(
        item if isinstance(item, str) else "<image>" for item in content_items
    )


def is_retryable_api_error(error):
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    return status_code in {408, 409, 429} or status_code >= 500


class IdentifierBase:
    def __init__(
        self,
        name,
        log_file=None,
        decoding_profile=DEFAULT_DECODING_PROFILE,
    ):
        self.name = name
        self.log_file = log_file
        self.decoding_profile = decoding_profile
        self.evaluation_protocol_version = evaluation_protocol_version(
            decoding_profile
        )
        self.generation_parameters = None

    def identify(self, content_items, num_candidates):
        self._log_prompt(content_items)
        text, response_metadata = self._generate(
            content_items,
            num_candidates,
        )
        text = text.strip()
        choice = parse_choice(text)
        valid = 1 <= choice <= num_candidates
        if not valid:
            choice = -2 if choice > 0 else -1

        reply = Identification(
            text=text,
            choice=choice,
            valid=valid,
            response_metadata=response_metadata,
        )
        self._log_response(reply)
        return reply

    def _generate(self, content_items, num_candidates):
        raise NotImplementedError

    def _log_prompt(self, content_items):
        if not self.log_file:
            return
        with open(self.log_file, "a") as f:
            f.write("-" * 40 + "\n")
            f.write(summarize_content(content_items))
            f.write("\n" + "-" * 40 + "\n")

    def _log_response(self, reply):
        if not self.log_file:
            return
        with open(self.log_file, "a") as f:
            f.write(f"Response:\n\n{reply.text}\n\n")


class RandomIdentifier(IdentifierBase):
    def __init__(
        self,
        log_file=None,
        decoding_profile=DEFAULT_DECODING_PROFILE,
    ):
        super().__init__("random", log_file, decoding_profile)

    def _generate(self, content_items, num_candidates):
        choice = random.randint(1, num_candidates)
        return f"Thought: random baseline\nChoice: [{choice}]", {}


class OpenAIIdentifier(IdentifierBase):
    def __init__(
        self,
        model,
        log_file=None,
        decoding_profile=DEFAULT_DECODING_PROFILE,
    ):
        super().__init__(model, log_file, decoding_profile)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Run `source scripts/setup_env.sh` "
                "or export it before starting evaluation."
            )

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("BASE_URL")
            or "https://api.openai.com/v1"
        )

        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.generation_parameters = generation_parameters_for_profile(
            decoding_profile
        )

    def _generate(self, content_items, num_candidates):
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": self._to_openai_content(content_items)},
        ]

        for _ in range(50):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.generation_parameters,
                )
                choice = completion.choices[0]
                usage = getattr(completion, "usage", None)
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump(mode="json")
                elif hasattr(usage, "dict"):
                    usage = usage.dict()
                return (choice.message.content or ""), {
                    "response_id": getattr(completion, "id", None),
                    "response_model": getattr(completion, "model", None),
                    "response_created": getattr(completion, "created", None),
                    "system_fingerprint": getattr(
                        completion,
                        "system_fingerprint",
                        None,
                    ),
                    "service_tier": getattr(completion, "service_tier", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "token_usage": usage,
                }
            except Exception as exc:
                if not is_retryable_api_error(exc):
                    raise RuntimeError(
                        "Non-retryable API request error. The evaluator did "
                        "not alter the decoding profile or retry with different "
                        "generation parameters."
                    ) from exc
                print("Exception:", exc)
                time.sleep(5)

        raise RuntimeError("Failed to get response from the model.")

    def _to_openai_content(self, content_items):
        payload = []
        for item in content_items:
            if isinstance(item, str):
                payload.append({"type": "text", "text": item})
            elif isinstance(item, np.ndarray):
                payload.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(item),
                            "detail": IMAGE_DETAIL,
                        },
                    }
                )
            else:
                raise TypeError(f"Unsupported prompt item: {type(item)!r}")
        return payload


def create_identifier(
    model,
    log_file=None,
    decoding_profile=DEFAULT_DECODING_PROFILE,
):
    if model == "random":
        return RandomIdentifier(log_file, decoding_profile)
    return OpenAIIdentifier(model, log_file, decoding_profile)
