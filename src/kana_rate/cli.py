import argparse
import glob
import os

from .parsing import merge_intervals, parse_ass, parse_srt, strip_nonspoken
from .reading import KanaReader


def _analyze_items(items, reader: KanaReader, use_mora: bool):
    texts = []
    intervals = []
    for start, end, text in items:
        if not text.strip():
            continue
        texts.append(text)
        intervals.append((start, end))

    total_units = 0
    for text in texts:
        text = strip_nonspoken(text)
        if not text.strip():
            continue
        reading = reader.to_kana(text)
        if use_mora:
            total_units += reader.count_mora(reading)
        else:
            total_units += reader.count_kana(reading)

    merged = merge_intervals(intervals)
    total_ms = sum(e - s for s, e in merged)
    minutes = total_ms / 1000.0 / 60.0 if total_ms > 0 else 0.0
    rate = (total_units / minutes) if minutes > 0 else 0.0
    return total_units, minutes, rate


def _collect_files(path: str):
    if os.path.isfile(path):
        return [path]
    escaped = glob.escape(path)
    srt_files = sorted(glob.glob(os.path.join(escaped, "*.srt")))
    if srt_files:
        return srt_files
    ass_files = sorted(glob.glob(os.path.join(escaped, "*.ass")))
    return ass_files


def main():
    parser = argparse.ArgumentParser(description="Compute mora-per-minute from subtitles.")
    parser.add_argument("path", help="Subtitle file or directory")
    parser.add_argument("--kana", action="store_true", help="Compute kana-per-minute instead of mora-per-minute")
    args = parser.parse_args()

    files = _collect_files(args.path)
    if not files:
        print("No .srt or .ass files found.")
        return

    reader = KanaReader()
    use_mora = not args.kana
    total_units = 0
    total_minutes = 0.0

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".srt":
            items = parse_srt(path)
        else:
            items = parse_ass(path)
        units, minutes, rate = _analyze_items(items, reader, use_mora)
        total_units += units
        total_minutes += minutes
        unit = "mora" if use_mora else "kana"
        print(f"{os.path.basename(path)}\t{units} {unit}\t{minutes:.2f} min\t{rate:.2f} {unit}/min")

    total_rate = (total_units / total_minutes) if total_minutes > 0 else 0.0
    unit = "mora" if use_mora else "kana"
    print(f"TOTAL\t{total_units} {unit}\t{total_minutes:.2f} min\t{total_rate:.2f} {unit}/min")


if __name__ == "__main__":
    main()
