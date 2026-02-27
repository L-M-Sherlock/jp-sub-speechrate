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
KANA_TILDE_RE = re.compile(r"(?<=[\u3040-\u309F\u30A0-\u30FF])[～〜]+")
SOKUON_RE = re.compile(r"[っッ]")
SMALL_KANA = set("ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ")

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
    text = KANA_TILDE_RE.sub("", text)
    text = NON_JP_RE.sub("", text)
    return text


class KanaReader:
    def __init__(self):
        self._tokenizer = dictionary.Dictionary().create()
        self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def to_kana(self, text: str, strip_sokuon: bool = True) -> str:
        parts = []
        text = _jiten_preprocess(text)
        for token in self._tokenizer.tokenize(text, self._mode):
            pos = token.part_of_speech()
            # Skip whitespace tokens (including full-width space) before counting.
            if pos and pos[0] == "空白":
                continue
            if pos and pos[0] in ("記号", "補助記号"):
                continue
            reading = token.reading_form()
            if reading == "*":
                reading = token.surface()
            if strip_sokuon:
                reading = SOKUON_RE.sub("", reading)
            parts.append(reading)
        return "".join(parts)

    @staticmethod
    def count_kana(text: str) -> int:
        return len(KANA_RE.findall(text))

    @staticmethod
    def count_mora(text: str) -> int:
        count = 0
        for ch in text:
            if ch in SMALL_KANA:
                continue
            if ch == "ー":
                continue
            if ch in ("っ", "ッ"):
                continue
            code = ord(ch)
            if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
                count += 1
        return count


def total_kana_count(reader: KanaReader, texts: Iterable[str]) -> int:
    count = 0
    for t in texts:
        kana = reader.to_kana(t)
        count += reader.count_kana(kana)
    return count
