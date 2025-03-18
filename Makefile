.PHONY: setup install test clean verify setup-env check-config

# Detect the operating system
ifeq ($(OS),Windows_NT)
	VENV_BIN := .venv\Scripts
	PYTHON := python
else
	VENV_BIN := .venv/bin
	PYTHON := python3
endif

setup: clean  ## Create virtual environment and install dependencies
	$(PYTHON) -m venv .venv
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e .
	@echo "\nSetup complete! To activate your virtual environment:"
	@echo "- Unix/MacOS: source .venv/bin/activate"
	@echo "- Windows: .venv\\Scripts\\activate"
	@echo "\nThen run: make configure"

configure:  ## Configure API keys and settings
	$(VENV_BIN)/python setup.py --configure

verify:  ## Verify the setup is working
	$(VENV_BIN)/python examples/verify_setup.py

install:  ## Install package in development mode
	$(VENV_BIN)/pip install -e .

test:  ## Run tests
	$(VENV_BIN)/pytest

clean:  ## Clean up generated files
	rm -rf .venv build dist *.egg-info __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup-env:
	@echo "Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
		echo "Please edit .env and add your API keys"; \
	else \
		echo ".env file already exists"; \
	fi

check-config:
	@echo "Checking configuration..."
	@python -c "from agents.config import get_openai_api_key, get_openai_org_id, get_anthropic_api_key, load_config; \
		print('\nConfiguration Status:'); \
		try: \
			api_key = get_openai_api_key(); \
			print('✓ OpenAI API key configured'); \
		except ValueError as e: \
			print('✗ OpenAI API key error:', str(e)); \
		try: \
			org_id = get_openai_org_id(); \
			if org_id: \
				print('✓ OpenAI organization ID configured'); \
			else: \
				print('! OpenAI organization ID not configured (optional)'); \
		except ValueError as e: \
			print('✗ OpenAI organization ID error:', str(e)); \
		anth_key = get_anthropic_api_key(); \
		if anth_key: \
			print('✓ Anthropic API key configured'); \
		else: \
			print('! Anthropic API key not configured (optional)'); \
		config = load_config(); \
		print('\nPoints Budget:'); \
		for phase, points in config['points_budget'].items(): \
			print(f'  {phase}: {points}');" 