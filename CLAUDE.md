# Working in this repo

- Start new branches from an up-to-date main: `git fetch` first, then branch from `origin/main`.
  PRs are merged on GitHub, so the local `main` is often behind.
- Run tests with `PYTHONPATH=. uv run pytest`.
