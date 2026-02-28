import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _trim_iqr(values: list[float]) -> list[float]:
    if len(values) < 4:
        return values
    sorted_vals = sorted(values)
    q1 = _percentile(sorted_vals, 25)
    q3 = _percentile(sorted_vals, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return values
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [v for v in values if lower <= v <= upper]


def _line_entries(items, reader: KanaReader, unit: str) -> list[tuple[int, int, int, float]]:
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
            count = reader.count_mora(reading)
        elif unit == "syllable":
            count = reader.count_syllable(reading)
        else:
            count = reader.count_kana(reading)
        if count <= 0:
            continue
        rate = count / (duration_ms / 1000.0 / 60.0)
        entries.append((start, end, count, rate))
    return entries


def _episode_rate(items, reader: KanaReader, unit: str, trim_outliers: bool) -> float:
    entries = _line_entries(items, reader, unit)
    if not entries:
        return 0.0

    if trim_outliers:
        rates = sorted(e[3] for e in entries)
        q1 = _percentile(rates, 25)
        q3 = _percentile(rates, 75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            entries = [e for e in entries if lower <= e[3] <= upper]
    if not entries:
        return 0.0

    total = sum(e[2] for e in entries)
    merged = merge_intervals([(e[0], e[1]) for e in entries])
    total_ms = sum(e - s for s, e in merged)
    minutes = total_ms / 1000.0 / 60.0 if total_ms > 0 else 0.0
    return (total / minutes) if minutes > 0 else 0.0


def _line_rates(items, reader: KanaReader, unit: str) -> list[float]:
    return [e[3] for e in _line_entries(items, reader, unit)]


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
        description="Visualize per-episode and per-show subtitle rate distributions."
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
        "--granularity",
        choices=["episode", "line"],
        default="episode",
        help="Distribution granularity: per episode or per subtitle line (default: episode)",
    )
    parser.add_argument(
        "--trim-outliers",
        action="store_true",
        help="Trim outliers using IQR before computing distributions",
    )
    parser.add_argument(
        "--include-subtitle-backup",
        action="store_true",
        help="Include SubtitleBackup folders",
    )
    parser.add_argument(
        "--out",
        default="rate_distributions",
        help="Output directory for per-show images (default: rate_distributions)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    show_dirs = _collect_show_dirs(root, not args.include_subtitle_backup)
    if not show_dirs:
        print("No subtitle folders found.")
        return

    reader = KanaReader()

    show_rates: dict[str, list[float]] = {}
    for d in show_dirs:
        rates = []
        for fname in sorted(d.iterdir()):
            if fname.suffix.lower() not in (".srt", ".ass"):
                continue
            items = _parse_items(fname)
            if args.granularity == "episode":
                rate = _episode_rate(items, reader, args.unit, args.trim_outliers)
                if rate > 0:
                    rates.append(rate)
            else:
                rates.extend(_line_rates(items, reader, args.unit))
        if rates:
            if args.granularity == "line" and args.trim_outliers:
                rates = _trim_iqr(rates)
            show_rates[d.name] = rates

    if not show_rates:
        print("No valid subtitle entries found.")
        return

    plt.rcParams["font.family"] = "Hiragino Sans"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def safe_name(name: str) -> str:
        # Preserve Unicode (including CJK). Only replace path-unsafe characters.
        return "".join("_" if ch in ("/", "\0", ":") else ch for ch in name).strip()

    for show, rates in show_rates.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        ax.hist(rates, bins=20)
        if args.granularity == "episode":
            subtitle = f"{len(rates)} eps"
        else:
            subtitle = f"{len(rates)} lines"
        ax.set_title(f"{show} ({subtitle}) - {args.unit}/min distribution")
        ax.set_xlabel(f"{args.unit}/min")
        ax.set_ylabel("Episode count")
        filename = safe_name(show) + f"_{args.unit}_{args.granularity}.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
