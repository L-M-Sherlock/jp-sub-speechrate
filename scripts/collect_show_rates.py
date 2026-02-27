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
        reading = reader.to_kana(text)
        if unit == "mora":
            count = reader.count_mora(reading)
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
        description="Compute per-show mora/kana rates recursively under a root directory."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to scan for subtitle folders (default: current directory)",
    )
    parser.add_argument(
        "--unit",
        choices=["mora", "kana"],
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
        for fname in sorted(d.iterdir()):
            if fname.suffix.lower() not in (".srt", ".ass"):
                continue
            items = _parse_items(fname)
            units, minutes, _ = _analyze_items(items, reader, args.unit, trim_outliers)
            total_units += units
            total_minutes += minutes
        if total_minutes > 0:
            rate = total_units / total_minutes
            rows.append((d.name, total_units, total_minutes, rate))

    if not rows:
        print("No valid subtitle entries found.")
        return

    unit_label = "MORA" if args.unit == "mora" else "KANA"
    print(f"| DIR | {unit_label} | MIN | RATE |")
    print("| --- | --- | --- | --- |")
    for name, units, minutes, rate in sorted(rows, key=lambda r: r[3]):
        print(f"| {name} | {units} | {minutes:.2f} | {rate:.2f} |")


if __name__ == "__main__":
    main()
