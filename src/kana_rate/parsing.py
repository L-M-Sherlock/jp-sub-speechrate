import re
from typing import Iterable, List, Tuple

import pysrt


TAG_RE = re.compile(r"\{[^}]*\}|<[^>]*>")
BRACKET_SEG_RE = re.compile(r"(\([^\)]*\)|（[^）]*）|\[[^\]]*\]|【[^】]*】)")
ONLY_BRACKETS_RE = re.compile(r"^\s*(?:" + BRACKET_SEG_RE.pattern + r"\s*)+$")
PREFIX_BRACKET_RE = re.compile(r"^\s*" + BRACKET_SEG_RE.pattern + r"\s*")
MUSIC_ONLY_RE = re.compile(r"^[\s♪～〜ー—…・･]+$")
KANA_ONLY_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FFー・･\s]+$")
CUE_WORDS = [
    "BGM",
    "SE",
    "効果音",
    "拍手",
    "歓声",
    "ざわ",
    "ざわざわ",
    "笑",
    "泣",
    "息",
    "ため息",
    "鼻歌",
    "ドア",
    "足音",
    "電話",
    "着信",
    "通知",
    "メール",
    "チャイム",
    "鈴",
    "ベル",
    "雷",
    "雨",
    "風",
]
CUE_RE = re.compile("|".join(re.escape(w) for w in CUE_WORDS), re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    text = TAG_RE.sub("", text)
    return text


def strip_nonspoken(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if MUSIC_ONLY_RE.match(line):
            continue
        if ONLY_BRACKETS_RE.match(line):
            continue

        # Remove leading speaker labels like "（柚子）".
        for _ in range(3):
            m = PREFIX_BRACKET_RE.match(line)
            if not m:
                break
            line = line[m.end() :].lstrip(" 　/／・-–—:：")

        # Drop bracketed segments that look like SFX or non-spoken cues.
        def _strip_cue(match: re.Match) -> str:
            seg = match.group(0)
            inner = seg[1:-1].strip()
            if CUE_RE.search(seg):
                return ""
            if inner and KANA_ONLY_RE.match(inner):
                return ""
            return seg

        line = BRACKET_SEG_RE.sub(_strip_cue, line).strip()
        if not line:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    intervals = [i for i in intervals if i[1] > i[0]]
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def merge_duplicate_items(
    items: Iterable[Tuple[int, int, str]], max_gap_ms: int = 0
) -> List[Tuple[int, int, str]]:
    grouped: dict[str, list[Tuple[int, int]]] = {}
    for start, end, text in items:
        grouped.setdefault(text, []).append((start, end))

    merged_items: list[Tuple[int, int, str]] = []
    for text, spans in grouped.items():
        spans.sort()
        cur_start, cur_end = spans[0]
        for start, end in spans[1:]:
            if start <= cur_end + max_gap_ms:
                cur_end = max(cur_end, end)
            else:
                merged_items.append((cur_start, cur_end, text))
                cur_start, cur_end = start, end
        merged_items.append((cur_start, cur_end, text))

    merged_items.sort(key=lambda x: (x[0], x[1], x[2]))
    return merged_items


def parse_srt(path: str) -> List[Tuple[int, int, str]]:
    subs = pysrt.open(path)
    items = []
    for sub in subs:
        text = clean_text(sub.text or "")
        items.append((sub.start.ordinal, sub.end.ordinal, text))
    return merge_duplicate_items(items, max_gap_ms=200)


def _parse_ass_time(ts: str) -> int:
    # ASS time format: H:MM:SS.CC
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return 0
    h = int(parts[0])
    m = int(parts[1])
    s_cs = parts[2].split(".")
    s = int(s_cs[0])
    cs = int(s_cs[1]) if len(s_cs) > 1 else 0
    return ((h * 3600 + m * 60 + s) * 1000) + (cs * 10)


def parse_ass(path: str) -> List[Tuple[int, int, str]]:
    items = []
    in_events = False
    event_format = None
    idx_start = idx_end = idx_text = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[Events]"):
                in_events = True
                event_format = None
                idx_start = idx_end = idx_text = None
                continue
            if stripped.startswith("[") and not stripped.startswith("[Events]"):
                in_events = False
                continue
            if not in_events:
                continue

            if stripped.startswith("Format:"):
                _, rest = line.split(":", 1)
                event_format = [f.strip() for f in rest.split(",")]
                idx_start = event_format.index("Start") if "Start" in event_format else None
                idx_end = event_format.index("End") if "End" in event_format else None
                idx_text = event_format.index("Text") if "Text" in event_format else None
                continue

            if stripped.startswith("Dialogue:"):
                if event_format is None or idx_start is None or idx_end is None or idx_text is None:
                    continue
                _, rest = line.split(":", 1)
                fields = rest.lstrip().split(",", maxsplit=len(event_format) - 1)
                if len(fields) <= max(idx_start, idx_end, idx_text):
                    continue
                start = _parse_ass_time(fields[idx_start])
                end = _parse_ass_time(fields[idx_end])
                text = clean_text(fields[idx_text])
                items.append((start, end, text))

    return merge_duplicate_items(items, max_gap_ms=200)
