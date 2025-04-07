# Tests

This directory contains tests for the MLOps pipeline.

## Running Tests

Run all tests:

```bash
python -m pytest
```

Run with coverage report:

```bash
python -m pytest --cov=.
```

Run specific test file:

```bash
python -m pytest tests/test_model_training.py
```

## Test Structure

- `test_model_training.py` - Tests for the model training pipeline

## CI/CD Integration

These tests are run automatically in the GitHub Actions CI/CD pipeline:

1. On each pull request to main
2. On each push to main

See `.github/workflows/mlops-pipeline.yml` for details.

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. Create a new test file in this directory
2. Name it `test_<module_name>.py`
3. Use the unittest or pytest framework
4. Include tests for both normal operation and edge cases/error handling

## Test Best Practices

- Keep tests independent from each other
- Use appropriate test fixtures
- Clean up after tests (especially for Kubernetes resources)
- Mock external services when appropriate
- Include both unit tests and integration tests 