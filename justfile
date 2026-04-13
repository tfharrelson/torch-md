fmt:
  uv run ruff format .

check:
  uv run ruff check .
  uv run pyright .

test:
  uv run pytest tests/

