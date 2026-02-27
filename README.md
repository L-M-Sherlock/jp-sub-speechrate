# jp-sub-speechrate

Compute **mora-per-minute** from subtitle files using Japanese reading conversion. This tool converts kanji to kana (via SudachiPy) before counting, so the rate reflects spoken Japanese rather than just visible kana.

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
uv run src/jp_sub_speechrate/cli.py ./subtitles

# Analyze a single file
uv run src/jp_sub_speechrate/cli.py ./file.srt
```

## Usage
```
jsub-rate <path> [--kana] [--include-outliers]
```
- `<path>` can be a file or a directory.
- If `<path>` is a directory, the tool processes all `.srt` files first. If no `.srt` are found, it falls back to `.ass`.
- If you are not installing the package, run `uv run src/jp_sub_speechrate/cli.py <path>` instead.
- By default the tool reports **mora/min**. Use `--kana` to report kana/min instead.
- By default per-line rate outliers are trimmed (IQR). Use `--include-outliers` to keep them.

Output format:
```
<filename>\t<count> <unit>\t<minutes> min\t<rate> <unit>/min
TOTAL\t<count> <unit>\t<minutes> min\t<rate> <unit>/min
```

## Visualization
The repository includes a plotting script to visualize rate distributions:
```bash
uv run scripts/visualize_rates.py --root /path/to/subtitles --out rate_distributions_lines --granularity line
```
- Use `--granularity episode` for per-episode distributions.
- Add `--trim-outliers` to apply IQR trimming before plotting.
- Use `--unit kana` to plot kana/min instead of mora/min.

## Per-show Summary (Recursive)
Compute a per-show summary table by scanning a root directory recursively (Markdown output, sorted by rate):
```bash
uv run scripts/collect_show_rates.py --root /path/to/subtitles
```

## How the mora count is computed
**What is a mora?** A mora is a timing unit in Japanese phonology (roughly a beat). For example, small kana combine with the preceding mora: 「きゃ」 counts as 1 mora, so 「きゃく」 is 2 mora (きゃ・く), and 「しゅっぱつ」 is 4 mora (しゅ・っ・ぱ・つ).
**How this project’s mora differs from the linguistic definition:** we approximate mora counts from Sudachi readings and subtitle timing. This means we count mora from the kana reading after normalization (e.g., symbols/whitespace removed, sokuon stripped, long vowels counted), and we do not model prosody, pauses, or coarticulation. The result is a practical “subtitle mora rate,” not a phonetic ground truth.
1. Subtitle text is cleaned:
   - ASS override tags `{...}` and HTML tags are removed.
   - `\N` line breaks are normalized.
   - Non-spoken cues are stripped (e.g., speaker labels like `（柚子）`, pure SFX lines, and music-only symbols).
2. Text is normalized for analysis:
   - ASCII digits are converted to full-width digits.
   - Kana tildes (e.g., `～`) attached to kana are removed.
   - Non-Japanese characters are dropped before tokenization.
3. SudachiPy converts each line to kana readings (katakana).
   - Whitespace and symbol tokens are ignored.
   - Sokuon (`っ`/`ッ`) is removed before counting.
4. Mora are counted from the kana reading.
   - Small kana are treated as part of the preceding mora.
   - Long vowel `ー` counts as one mora.
5. Subtitle time spans are merged to avoid double-counting overlapping lines.
6. By default, per-line rate outliers are trimmed (IQR) before computing totals. Use `--include-outliers` to keep them.

## Supported subtitle formats
- **SRT**: parsed via `pysrt`.
- **ASS/SSA**: parsed by reading `Dialogue:` lines from the `[Events]` section.

## Files and structure
```
./src/jp_sub_speechrate/
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
