# Agent Guidelines

## Pre-Push Requirements

Before every push to GitHub, the following commands must be executed and pass:

```sh
just fmt
just check
just test
```

- `just fmt` - Formats all code with `ruff format`.
- `just check` - Runs `ruff check` (linting) and `pyright` (type checking). Both must report zero errors.
- `just test` - Runs the full test suite with `pytest`. All tests must pass.

Do not push if any of these commands fail. Fix all issues first.
