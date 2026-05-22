# Development Guide

## Setup

pip install -r requirements-dev.txt

## Testing

pytest tests/ -v

pytest tests/ --cov=src --cov=api

## Code Quality

black src/ api/ tests/

ruff check src/ api/ tests/

mypy src/ api/

## Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Push and create PR
