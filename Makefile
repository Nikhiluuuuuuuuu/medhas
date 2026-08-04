# Medhas — production Makefile
# Common tasks for local dev, testing, and container builds.

PYTHON     ?= python3
VENV       ?= .venv
PYTHONPATH := $(CURDIR)
export PYTHONPATH

# Postgres test database wiring (override on the command line if needed).
export POSTGRES_USER  ?= agent_user
export POSTGRES_PASSWORD ?= agent_password
export POSTGRES_HOST  ?= 127.0.0.1
export POSTGRES_PORT  ?= 5432
export POSTGRES_DB    ?= medhas_test

.PHONY: help setup venv install install-dev deps test test-online lint fmt \
        run server docker-build docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create a local virtualenv
	$(PYTHON) -m venv $(VENV)

install:  ## Install runtime deps (provider-agnostic)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev: install  ## Install runtime + dev deps
	$(PYTHON) -m pip install -e ".[dev]"

deps: install-dev  ## Alias for install-dev

test:  ## Run the full pytest suite (needs Postgres + LLM key)
	$(PYTHON) -m pytest tests/ -q

test-online:  ## Run only the online (LLM-dependent) subset
	$(PYTHON) -m pytest tests/ -q -k "online or cognition or generalization or engine" --timeout=300

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check medhas/ tests/ main.py server.py

fmt:  ## Auto-format with ruff
	$(PYTHON) -m ruff format medhas/ tests/ main.py server.py

run:  ## Run the deterministic production test suite (CLI: medhas-test)
	$(PYTHON) -m main

server:  ## Run the FastAPI server (CLI: medhas-server)
	$(PYTHON) -m server

docker-build:  ## Build the container image
	docker build -t medhas:latest .

docker-up:  ## Start Postgres + app via compose
	docker compose up -d --build

docker-down:  ## Tear down compose services
	docker compose down

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage build dist *.egg-info
