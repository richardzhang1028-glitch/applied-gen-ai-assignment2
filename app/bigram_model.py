"""A compact bigram language model for the Module 3 API activity."""

from collections import Counter, defaultdict
import random
import re


class BigramModel:
    """Generate short sequences using word-to-word transition counts."""

    def __init__(self, corpus: list[str]):
        transitions: dict[str, Counter[str]] = defaultdict(Counter)
        for text in corpus:
            words = re.findall(r"\b[\w']+\b", text.lower())
            for current, following in zip(words, words[1:]):
                transitions[current][following] += 1
        self.transitions = dict(transitions)

    def generate_text(self, start_word: str, length: int) -> str:
        """Return up to ``length`` words, beginning with ``start_word``."""
        current = start_word.strip().lower()
        if not current:
            raise ValueError("start_word must not be blank")

        generated = [current]
        for _ in range(length - 1):
            next_words = self.transitions.get(current)
            if not next_words:
                break
            words = list(next_words)
            weights = list(next_words.values())
            current = random.choices(words, weights=weights, k=1)[0]
            generated.append(current)

        return " ".join(generated)
