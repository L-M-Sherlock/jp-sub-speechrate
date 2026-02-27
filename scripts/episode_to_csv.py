import argparse
import csv
from pathlib import Path

from jp_sub_speechrate.parsing import parse_ass, parse_srt, strip_nonspoken
from jp_sub_speechrate.reading import KanaReader


def _format_ms(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _parse_items(path: Path):
    if path.suffix.lower() == ".srt":
        return parse_srt(str(path))
    if path.suffix.lower() == ".ass":
        return parse_ass(str(path))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Export per-line subtitle rates for a single episode to CSV."
    )
    parser.add_argument("input", help="Subtitle file (.srt or .ass)")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument(
        "--unit",
        choices=["mora", "kana"],
        default="mora",
        help="Rate unit to compute (default: mora)",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")
    if src.suffix.lower() not in (".srt", ".ass"):
        raise SystemExit("Input must be .srt or .ass")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    items = _parse_items(src)
    reader = KanaReader()

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["START", "END", "DURATION (s)", args.unit.upper(), "RATE", "TEXT"])
        for start, end, text in items:
            if not text.strip():
                continue
            cleaned = strip_nonspoken(text)
            if not cleaned.strip():
                continue
            duration_ms = end - start
            if duration_ms <= 0:
                continue
            reading = reader.to_kana(cleaned)
            if args.unit == "mora":
                count = reader.count_mora(reading)
            else:
                count = reader.count_kana(reading)
            if count <= 0:
                continue
            duration_s = duration_ms / 1000.0
            rate = count / (duration_s / 60.0)
            writer.writerow(
                [
                    _format_ms(start),
                    _format_ms(end),
                    f"{duration_s:.3f}",
                    count,
                    f"{rate:.2f}",
                    cleaned.replace("\n", " / "),
                ]
            )

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
