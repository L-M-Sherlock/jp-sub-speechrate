import argparse
import glob
import os
import sys

try:
    from .parsing import merge_intervals, parse_ass, parse_srt, strip_nonspoken
    from .reading import KanaReader
except ImportError:
    # Allow running as a script: `uv run src/jp_sub_speechrate/cli.py ...`
    pkg_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.dirname(pkg_dir))
    from jp_sub_speechrate.parsing import merge_intervals, parse_ass, parse_srt, strip_nonspoken
    from jp_sub_speechrate.reading import KanaReader


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _analyze_items(items, reader: KanaReader, unit: str, trim_outliers: bool):
    entries = []
    for start, end, text in items:
        if not text.strip():
            continue
        text = strip_nonspoken(text)
        if not text.strip():
            continue
        duration_ms = end - start
        if duration_ms <= 0:
            continue
        strip_sokuon = unit != "syllable"
        reading = reader.to_kana(text, strip_sokuon=strip_sokuon)
        if unit == "mora":
            units = reader.count_mora(reading)
        elif unit == "syllable":
            units = reader.count_syllable(reading)
        else:
            units = reader.count_kana(reading)
        if units <= 0:
            continue
        rate = units / (duration_ms / 1000.0 / 60.0)
        entries.append((start, end, units, rate))

    if not entries:
        return 0, 0.0, 0.0

    if trim_outliers and len(entries) >= 4:
        rates = sorted(e[3] for e in entries)
        q1 = _percentile(rates, 25)
        q3 = _percentile(rates, 75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            entries = [e for e in entries if lower <= e[3] <= upper]

    if not entries:
        return 0, 0.0, 0.0

    total_units = sum(e[2] for e in entries)
    intervals = [(e[0], e[1]) for e in entries]
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
    parser = argparse.ArgumentParser(description="Compute mora/kana/syllable rates from subtitles.")
    parser.add_argument("path", help="Subtitle file or directory")
    parser.add_argument("--kana", action="store_true", help="Compute kana-per-minute instead of mora-per-minute")
    parser.add_argument(
        "--unit",
        choices=["mora", "kana", "syllable"],
        help="Rate unit to compute (overrides --kana when provided)",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include per-line rate outliers (by default they are trimmed using IQR)",
    )
    args = parser.parse_args()

    files = _collect_files(args.path)
    if not files:
        print("No .srt or .ass files found.")
        return

    reader = KanaReader()
    if args.unit:
        unit = args.unit
    else:
        unit = "kana" if args.kana else "mora"
    total_units = 0
    total_minutes = 0.0

    trim_outliers = not args.include_outliers
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".srt":
            items = parse_srt(path)
        else:
            items = parse_ass(path)
        units, minutes, rate = _analyze_items(items, reader, unit, trim_outliers)
        total_units += units
        total_minutes += minutes
        print(f"{os.path.basename(path)}\t{units} {unit}\t{minutes:.2f} min\t{rate:.2f} {unit}/min")

    total_rate = (total_units / total_minutes) if total_minutes > 0 else 0.0
    print(f"TOTAL\t{total_units} {unit}\t{total_minutes:.2f} min\t{total_rate:.2f} {unit}/min")


if __name__ == "__main__":
    main()
