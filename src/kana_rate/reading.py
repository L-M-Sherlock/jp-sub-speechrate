import re
from typing import Iterable

from sudachipy import dictionary
from sudachipy import tokenizer as sudachi_tokenizer


KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
NON_JP_RE = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF21-\uFF3A\uFF41-\uFF5A\uFF10-\uFF19"
    r"\u3005\u3001-\u3003\u3008-\u3011\u3014-\u301F\uFF01-\uFF0F\uFF1A-\uFF1F\uFF3B-\uFF3F"
    r"\uFF5B-\uFF60\uFF62-\uFF65．\n…\u3000―\u2500()。！？「」）|]"
)

_FULLWIDTH_DIGITS = str.maketrans(
    {
        "0": "０",
        "1": "１",
        "2": "２",
        "3": "３",
        "4": "４",
        "5": "５",
        "6": "６",
        "7": "７",
        "8": "８",
        "9": "９",
    }
)


def _jiten_preprocess(text: str) -> str:
    # Mirror Jiten MorphologicalAnalyser.PreprocessText (minimal subset)
    text = text.replace("<", " ").replace(">", " ")
    text = text.translate(_FULLWIDTH_DIGITS)
    text = NON_JP_RE.sub("", text)
    return text


class KanaReader:
    def __init__(self):
        self._tokenizer = dictionary.Dictionary().create()
        self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def to_kana(self, text: str) -> str:
        parts = []
        text = _jiten_preprocess(text)
        for token in self._tokenizer.tokenize(text, self._mode):
            pos = token.part_of_speech()
            # Jiten strips ASCII spaces before Sudachi; keep full-width spaces.
            if pos and pos[0] == "空白" and token.surface() in (" ", "\t"):
                continue
            reading = token.reading_form()
            if reading == "*":
                reading = token.surface()
            parts.append(reading)
        return "".join(parts)

    @staticmethod
    def count_kana(text: str) -> int:
        return len(KANA_RE.findall(text))


def total_kana_count(reader: KanaReader, texts: Iterable[str]) -> int:
    count = 0
    for t in texts:
        kana = reader.to_kana(t)
        count += reader.count_kana(kana)
    return count
