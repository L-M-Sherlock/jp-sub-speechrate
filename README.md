# kana-rate

Compute **kana-per-minute** from subtitle files using Japanese reading conversion. This tool converts kanji to kana (via SudachiPy) before counting, so the rate reflects spoken Japanese rather than just visible kana.

## What it does
- Reads `.srt` and `.ass` subtitle files.
- Converts text to kana readings with SudachiPy.
- Counts kana characters and divides by **merged subtitle duration** (overlaps are merged).
- Reports per-file and total mora/minute by default (use `--kana` for kana/minute).

## Requirements
- Python 3.10+
- `uv` recommended for environment management

## Quickstart (uv)
```bash
cd kana-rate
# Analyze a directory (all .srt first, else .ass)
uv run src/kana_rate/cli.py ./subtitles

# Analyze a single file
uv run src/kana_rate/cli.py ./file.srt
```

## Usage
```
kana-rate <path> [--kana] [--trim-outliers]
```
- `<path>` can be a file or a directory.
- If `<path>` is a directory, the tool processes all `.srt` files first. If no `.srt` are found, it falls back to `.ass`.
- If you are not installing the package, run `uv run src/kana_rate/cli.py <path>` instead.
- By default the tool reports **mora/min**. Use `--kana` to report kana/min instead.
- Use `--trim-outliers` to drop per-line rate outliers (IQR) before computing totals.

Output format:
```
<filename>\t<count> <unit>\t<minutes> min\t<rate> <unit>/min
TOTAL\t<count> <unit>\t<minutes> min\t<rate> <unit>/min
```

## How the kana count is computed
1. Subtitle text is cleaned:
   - ASS override tags `{...}` and HTML tags are removed.
   - `\N` line breaks are normalized.
   - Non-spoken cues are stripped (e.g., speaker labels like `（柚子）`, pure SFX lines, and music-only symbols).
2. SudachiPy converts each line to kana readings (katakana).
3. Only kana characters (hiragana/katakana) are counted.
4. Subtitle time spans are merged to avoid double-counting overlapping lines.

## Supported subtitle formats
- **SRT**: parsed via `pysrt`.
- **ASS/SSA**: parsed by reading `Dialogue:` lines from the `[Events]` section.

## Files and structure
```
./src/kana_rate/
  cli.py        # CLI entry point
  parsing.py    # subtitle parsing and time merging
  reading.py    # SudachiPy conversion to kana
```

## Development notes
- SudachiPy `reading()` returns katakana. This is fine for counting kana characters.
- If a token has no reading (returns `*`), the surface form is used.
- Merged duration is computed in milliseconds and converted to minutes.

## Troubleshooting
- If SudachiPy dictionary is missing, ensure dependencies are available in your environment.
- If parsing fails on unusual ASS files, check the `[Events]` section formatting.

## TODO
- Allow halfwidth katakana (e.g., `ｶﾀｶﾅ`, `ﾊﾟﾝ`, `ｰ`) to pass preprocessing so they are counted correctly.

## License
This project is a utility script intended for internal use. Add a license if you plan to distribute it.
