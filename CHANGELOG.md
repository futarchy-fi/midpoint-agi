# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Setup script for easy development environment setup
- Verification tool to diagnose common setup issues
- Mock implementation for testing without an OpenAI API key
- Comprehensive README with setup and usage instructions

### Changed
- Reorganized project structure to follow Python best practices
  - Moved from flat `agents/` directory to proper `src/midpoint/agents/` package
  - Updated all imports to use the new package structure
  - Configured pyproject.toml for proper package installation
- Improved error handling for missing or invalid API keys
- Enhanced test repository setup script with better user guidance

### Fixed
- Import errors caused by conflicting packages
- Environment variables not being loaded properly
- Installation issues due to improper project structure 