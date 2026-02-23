from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Choice:
    value: str
    label: str


class NeedAnswer(Exception):
    """
    Raised by the engine when it needs a user answer.
    The view catches this and renders a question page.
    """
    def __init__(self, key: str, prompt: str, choices: List[Choice]):
        self.key = key
        self.prompt = prompt
        self.choices = choices
        super().__init__(prompt)


def make_key(prompt: str, choices: List[Tuple[str, str]]) -> str:
    vals = "|".join(v for v, _ in choices)
    return f"q::{prompt}::{vals}"


class SessionIO:
    def __init__(self, session: Dict[str, Any]):
        self.session = session
        self.session.setdefault("answers", {})
        self.session.setdefault("progress_snippets", [])
        self.session.setdefault("progress_inline", [])  # for inline snippets (title/desc/code)

    def get_answer(self, key: str) -> Optional[str]:
        return self.session["answers"].get(key)

    def set_answer(self, key: str, value: str) -> None:
        self.session["answers"][key] = value

    def push_snippet(self, snippet_key: str) -> None:
        self.session["progress_snippets"].append(snippet_key)

    def push_inline(self, title: str, desc: str, code: str) -> None:
        self.session["progress_inline"].append({"title": title, "desc": desc, "code": code})

    def reset(self) -> None:
        self.session["answers"] = {}
        self.session["progress_snippets"] = []
        self.session["progress_inline"] = []