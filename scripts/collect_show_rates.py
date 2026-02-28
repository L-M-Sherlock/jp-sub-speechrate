import argparse
from pathlib import Path

from jp_sub_speechrate.parsing import merge_intervals, parse_ass, parse_srt, strip_nonspoken
from jp_sub_speechrate.reading import KanaReader


def _parse_items(path: Path):
    if path.suffix.lower() == ".srt":
        return parse_srt(str(path))
    if path.suffix.lower() == ".ass":
        return parse_ass(str(path))
    return []


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
        strip_sokuon = unit == "kana"
        reading = reader.to_kana(text, strip_sokuon=strip_sokuon)
        if unit == "mora":
            count = reader.count_mora(reading)
        elif unit == "syllable":
            count = reader.count_syllable(reading)
        else:
            count = reader.count_kana(reading)
        if count <= 0:
            continue
        rate = count / (duration_ms / 1000.0 / 60.0)
        entries.append((start, end, count, rate))

    if trim_outliers and entries:
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


def _line_rates(items, reader: KanaReader, unit: str) -> list[tuple[float, float]]:
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
        strip_sokuon = unit == "kana"
        reading = reader.to_kana(text, strip_sokuon=strip_sokuon)
        if unit == "mora":
            count = reader.count_mora(reading)
        elif unit == "syllable":
            count = reader.count_syllable(reading)
        else:
            count = reader.count_kana(reading)
        if count <= 0:
            continue
        duration_s = duration_ms / 1000.0
        rate = count / (duration_s / 60.0)
        entries.append((rate, duration_s))
    return entries


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    pairs = sorted(pairs, key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    target = total_w / 2.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return v
    return pairs[-1][0]


def _collect_show_dirs(root: Path, exclude_subtitle_backup: bool) -> list[Path]:
    exts = {".srt", ".ass"}
    dirs = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        if exclude_subtitle_backup and "SubtitleBackup" in path.parts:
            continue
        dirs.add(path.parent)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-show mora/kana/syllable rates recursively under a root directory."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to scan for subtitle folders (default: current directory)",
    )
    parser.add_argument(
        "--unit",
        choices=["mora", "kana", "syllable"],
        default="mora",
        help="Rate unit to compute (default: mora)",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include per-line rate outliers (by default they are trimmed using IQR)",
    )
    parser.add_argument(
        "--include-subtitle-backup",
        action="store_true",
        help="Include SubtitleBackup folders",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    show_dirs = _collect_show_dirs(root, not args.include_subtitle_backup)
    if not show_dirs:
        print("No subtitle folders found.")
        return

    reader = KanaReader()
    trim_outliers = not args.include_outliers

    rows = []
    for d in show_dirs:
        total_units = 0
        total_minutes = 0.0
        line_rates = []
        for fname in sorted(d.iterdir()):
            if fname.suffix.lower() not in (".srt", ".ass"):
                continue
            items = _parse_items(fname)
            units, minutes, rate = _analyze_items(items, reader, args.unit, trim_outliers)
            total_units += units
            total_minutes += minutes
            line_rates.extend(_line_rates(items, reader, args.unit))
        if total_minutes > 0:
            rate = total_units / total_minutes
            if trim_outliers and len(line_rates) >= 4:
                rates_only = sorted(r for r, _ in line_rates)
                q1 = _percentile(rates_only, 25)
                q3 = _percentile(rates_only, 75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    line_rates = [(r, w) for r, w in line_rates if lower <= r <= upper]
            line_median_tw = _weighted_median(line_rates)
            rows.append((d.name, total_units, total_minutes, rate, line_median_tw))

    if not rows:
        print("No valid subtitle entries found.")
        return

    unit_label = "MORA" if args.unit == "mora" else "SYLLABLE" if args.unit == "syllable" else "KANA"
    print(f"| DIR | {unit_label} | MIN | RATE | LINE_MEDIAN_TW |")
    print("| --- | --- | --- | --- | --- |")
    for name, units, minutes, rate, line_median_tw in sorted(rows, key=lambda r: r[3]):
        print(f"| {name} | {units} | {minutes:.2f} | {rate:.2f} | {line_median_tw:.2f} |")


if __name__ == "__main__":
    main()
