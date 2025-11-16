ifneq (,$(wildcard ./.env))
    include .env
    export
endif

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
	rm -r dist/
	uv build
publish-test:
	uv publish --publish-url https://test.pypi.org/legacy/ --token $(TEST_PYPI_API_TOKEN)
publish-prod:
	uv publish --token $(PYPI_API_TOKEN)
