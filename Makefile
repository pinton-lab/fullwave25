install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
install-precommit:
	brew install pre-commit
install:
	uv sync
	uv run pre-commit install
install-dev:
	uv sync --dev
	uv run pre-commit install
install-examples:
	uv sync --extra examples
	uv run pre-commit install
test:
	uv run pytest
install-all-extras:
	uv sync --all-extras
	uv run pre-commit install
build:
	uv build
