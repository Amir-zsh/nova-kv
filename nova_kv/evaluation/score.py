"""Small, dependency-free scorers used by the paper wrappers."""
import re


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def exact_match(prediction: str, answers: list[str]) -> float:
    p = normalize(prediction)
    return float(any(p == normalize(a) for a in answers))


def contains_answer(prediction: str, answers: list[str]) -> float:
    p = normalize(prediction)
    return float(any(normalize(a) in p for a in answers))


def pass_at_1(results: list[bool]) -> float:
    return sum(map(bool, results)) / len(results) if results else 0.0
