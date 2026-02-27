# Repository Guidelines

## Project Structure & Module Organization
- `src/kana_rate/` contains the application code.
  - `cli.py` is the CLI entry point (`kana-rate`).
  - `parsing.py` handles subtitle parsing and time merging.
  - `reading.py` converts text to kana via SudachiPy.
- No `tests/` directory exists yet.
- No static assets or configuration files beyond `pyproject.toml`.

## Build, Test, and Development Commands
Use `uv` (recommended) for a local environment:
- `uv run src/kana_rate/cli.py ./subtitles` runs the CLI on a directory.
- `uv run src/kana_rate/cli.py ./file.srt` runs the CLI on a single file.
- Output is mora/min by default; pass `--kana` for kana/min.

Packaging is handled by `setuptools` per `pyproject.toml`. If you need a build artifact, use the `build` package in your environment.

## Coding Style & Naming Conventions
- Python 3.10+ only.
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions and modules (e.g., `kana_rate`, `parse_ass`).
- Keep CLI-facing behavior in `cli.py`, and put parsing/conversion logic in `parsing.py` or `reading.py`.
- No formatter or linter is configured yet; keep changes small and consistent with existing style.

## Testing Guidelines
- There is no test framework configured yet and no tests present.
- If adding tests, prefer `pytest` and place them under `tests/` with names like `test_parsing.py`.
- Document the test command you introduce in this file when you add it.

## Commit & Pull Request Guidelines
- This repository has no commit history yet, so no established convention exists.
- Use concise, imperative commit subjects (e.g., `Add ASS parsing for dialogue tags`).
- PRs should include a summary, how to run/verify changes, and any new CLI output examples when behavior changes.

## Configuration Notes
- SudachiPy requires a dictionary; if lookups fail, ensure dependencies are available in your environment.
- The CLI expects `.srt` or `.ass` inputs; directory mode prefers `.srt` and falls back to `.ass`.
