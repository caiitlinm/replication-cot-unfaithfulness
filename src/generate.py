import os
import re
from time import sleep
import traceback

from dotenv import load_dotenv

load_dotenv()

SEP = "\n\n###\n\n"

MODEL = os.environ.get("MODEL")
if not MODEL:
    raise RuntimeError(
        "MODEL env var is required. Set it in .env, e.g. MODEL=qwen/qwen3-8b"
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_TOKENS_COT = 4096
MAX_TOKENS_DIRECT = 256

TASKS = os.getenv("TASKS", "temporal_sequences,navigate").split(",")
EXAMPLES_PER_TASK = int(os.getenv("EXAMPLES_PER_TASK", "100"))


def _get_client():
    import openai
    return openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def get_reasoning(response):
    """Extract reasoning from an OpenRouter response object.

    Checks the separate reasoning field first, then falls back to
    parsing inline <think>...</think> tags from content.
    """
    msg = response.choices[0].message

    for attr in ("reasoning", "reasoning_content"):
        val = getattr(msg, attr, None)
        if val:
            return val

    content = msg.content or ""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


def get_content(response) -> str:
    """Extract the final answer content, stripping any inline <think> block."""
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()


class ModelMismatchError(Exception):
    pass


def add_retries(f):
    def wrap(*args, **kwargs):
        max_retries = 5
        num_retries = 0
        while True:
            try:
                return f(*args, **kwargs)
            except (KeyboardInterrupt, ModelMismatchError):
                raise
            except Exception as e:
                print("Error: ", traceback.format_exc(), "\nRetrying in", num_retries * 2, "seconds")
                if num_retries == max_retries:
                    traceback.print_exc()
                    return None
                num_retries += 1
                sleep(num_retries * 2)
    return wrap


@add_retries
def generate(prompt: str, model: str = None, max_tokens: int = MAX_TOKENS_COT,
             reasoning: bool = True) -> object:
    """Call OpenRouter. Returns the raw OpenAI-compatible response object."""
    model = model or MODEL
    client = _get_client()

    extra_body = {
        "reasoning": {"enabled": reasoning},
        "provider": {"require_parameters": True},
    }

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_body=extra_body,
    )

    resp_model = getattr(resp, 'model', None) or ''
    if resp_model and not resp_model.startswith(model.split(':')[0]):
        raise ModelMismatchError(
            f"Requested model '{model}' but got response from '{resp_model}'. "
            f"Aborting to prevent silent model substitution."
        )

    return resp


class Config:

    def __init__(self, task, **kwargs):
        self.task = task
        import datetime
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "model"):
            self.anthropic_model = False

    def __str__(self):
        base_str = self.time + "-" + self.task + "-" + self.model.replace("/", "_")
        for k, v in sorted(self.__dict__.items()):
            if k in ("time", "task", "model", "bias_text"):
                continue
            base_str = base_str + "-" + k.replace("_", "") + str(v).replace("-", "").replace('.json', '')
        return base_str
