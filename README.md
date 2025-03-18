# Midpoint

An advanced AGI system for recursive goal decomposition.

## Quick Start

1. Clone the repository
2. Copy `.env.example` to `.env` and add your OpenAI API key
3. Run `make setup` to install dependencies
4. Run `make test` to verify everything is working

```bash
# Clone the repository
git clone https://github.com/yourusername/midpoint.git
cd midpoint

# Setup virtual environment and install dependencies
make setup

# Activate virtual environment
# On Unix/MacOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Configure your API keys
make configure
```

## Prerequisites

- Python 3.9 or higher
- Make (optional, but recommended)
- Git

## Installation

### Using Make (Recommended)

1. Run the setup command:
   ```bash
   make setup
   ```
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up the development environment

2. Activate the virtual environment:
   ```bash
   # On Unix/MacOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. Configure your API keys:
   ```bash
   make configure
   ```

### Manual Installation

If you prefer not to use Make:

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   
   # On Unix/MacOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Configure your API keys:
   ```bash
   python setup.py --configure
   ```

## Configuration

API keys are stored securely in environment variables, which can be configured in two ways:

1. Create a `.env` file in your project directory (recommended for development)
   ```
   OPENAI_API_KEY=your-api-key
   OPENAI_ORG_ID=optional-org-id
   ```

2. Set environment variables directly (recommended for production/CI)
   ```bash
   export OPENAI_API_KEY=your-api-key
   export OPENAI_ORG_ID=optional-org-id
   ```

The `.env` file approach is preferred for development as it prevents accidental commits of API keys and allows for easy configuration switching.

### Supported API Providers

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)

## Development

### Available Make Commands

- `make setup` - Set up development environment
- `make setup-env` - Create .env file from .env.example
- `make check-config` - Verify API keys and configuration
- `make configure` - Configure API keys
- `make install` - Install package in development mode
- `make test` - Run tests
- `make clean` - Clean up generated files
- `make help` - Show available commands

### Running Tests

```bash
make test
```

### Project Structure

```
midpoint/
├── agents/           # Core agent implementation
├── docs/            # Documentation
├── examples/        # Example usage
├── tests/           # Test suite
├── .env.example     # Example environment variables
├── Makefile         # Development commands
├── pyproject.toml   # Project metadata and dependencies
└── README.md        # This file
```

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]

# API Key Configuration

Midpoint requires an OpenAI API key to function. There are two ways to configure it:

1. Create a `.env` file in the project directory (recommended for development)
   ```
   OPENAI_API_KEY=your-api-key
   OPENAI_ORG_ID=optional-org-id
   ```

2. Set environment variables directly (recommended for production/CI)
   ```bash
   export OPENAI_API_KEY=your-api-key
   export OPENAI_ORG_ID=optional-org-id
   ```

The `.env` file approach is preferred for development as it prevents accidental commits of API keys and allows for easy configuration switching.
