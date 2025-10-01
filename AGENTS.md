# RAG implementation Practice

A repo for me to practice my RAG skills.

## Environment & Dependencies

- Create or activate the local env with `uv venv`.
- Install Python dependencies via `uv pip install -r requirements.txt` (keeps the lockless requirements list in sync).
- Use `uv run <command>` to execute any project tooling so the virtual environment is respected.
- After installing new packages with `uv pip`, snapshot them with `uv pip freeze > requirements.txt`.

## Formatting & Quality Gates

- **Whenever you add or modify a Python file**, run these commands for each touched path:
  - `uv run black <path/to/file.py>`
  - `uv run ruff <path/to/file.py>`
  - `uv run mypy <path/to/file.py>`
- Keep diffs clean by running the formatter before committing and fix any lint or type errors immediately.

## Conventions & Layout

- Prefer module-level constants for prompts and config; avoid embedding secrets in code or committing `.env` values.

## Git Workflow

- Branch from `main` with conventional names like `feature/<topic>` or `fix/<issue>`.
- Ensure formatters, lint, and type-check pass before opening a PR.
- Commit messages should summarize intent in present tense.

## Tool Usage

- Use `webSearch` to gather current or rapidly changing information.
- When a user shares URLs, always call `fetchUrl` to inspect the linked content before acting on it.
- For third-party programming APIs or libraries, prefer `getCodeContextExa` to retrieve relevant documentation snippets.
