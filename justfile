fmt:
  uv run ruff format .

check:
  uv run ruff check .
  uv run pyright check .

test:
  uv run pytest tests/

