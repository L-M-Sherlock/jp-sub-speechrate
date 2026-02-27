import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kana_rate.parsing import merge_intervals, parse_ass, parse_srt, strip_nonspoken
from kana_rate.reading import KanaReader


def _parse_items(path: Path):
    if path.suffix.lower() == ".srt":
        return parse_srt(str(path))
    if path.suffix.lower() == ".ass":
        return parse_ass(str(path))
    return []


def _episode_rate(items, reader: KanaReader, unit: str) -> float:
    texts = []
    intervals = []
    for start, end, text in items:
        if not text.strip():
            continue
        text = strip_nonspoken(text)
        if not text.strip():
            continue
        texts.append(text)
        intervals.append((start, end))

    total = 0
    for text in texts:
        reading = reader.to_kana(text)
        if unit == "mora":
            total += reader.count_mora(reading)
        else:
            total += reader.count_kana(reading)

    merged = merge_intervals(intervals)
    total_ms = sum(e - s for s, e in merged)
    minutes = total_ms / 1000.0 / 60.0 if total_ms > 0 else 0.0
    return (total / minutes) if minutes > 0 else 0.0


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
        choices=["mora", "kana"],
        default="mora",
        help="Rate unit to compute (default: mora)",
    )
    parser.add_argument(
        "--include-subtitle-backup",
        action="store_true",
        help="Include SubtitleBackup folders",
    )
    parser.add_argument(
        "--out",
        default="rate_distributions.png",
        help="Output image path (default: rate_distributions.png)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    show_dirs = _collect_show_dirs(root, not args.include_subtitle_backup)
    if not show_dirs:
        print("No subtitle folders found.")
        return

    reader = KanaReader()

    show_rates: dict[str, list[float]] = {}
    all_episode_rates: list[float] = []
    for d in show_dirs:
        rates = []
        for fname in sorted(d.iterdir()):
            if fname.suffix.lower() not in (".srt", ".ass"):
                continue
            items = _parse_items(fname)
            rate = _episode_rate(items, reader, args.unit)
            if rate > 0:
                rates.append(rate)
                all_episode_rates.append(rate)
        if rates:
            show_rates[d.name] = rates

    if not show_rates:
        print("No valid subtitle entries found.")
        return

    shows = list(show_rates.keys())
    data = [show_rates[s] for s in shows]

    plt.rcParams["font.family"] = "Hiragino Sans"
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(shows) * 0.6), 10), constrained_layout=True)

    axes[0].boxplot(data, tick_labels=shows, showfliers=False)
    axes[0].set_title(f"Per-Show {args.unit.upper()}/min Distribution (episodes)")
    axes[0].set_ylabel(f"{args.unit}/min")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)

    axes[1].hist(all_episode_rates, bins=30)
    axes[1].set_title(f"Per-Episode {args.unit.upper()}/min Distribution (all shows)")
    axes[1].set_xlabel(f"{args.unit}/min")
    axes[1].set_ylabel("Episode count")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
