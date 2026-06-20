.PHONY: install test lint build run-api run-worker clean

# Dependency installation
install:
	poetry install

# Run pytest with coverage reporting and threshold check (fail if coverage < 80%)
test:
	poetry run pytest --cov=. --cov-report=term-missing --cov-fail-under=80

# Linting with ruff and type-checking with mypy
lint:
	poetry run ruff check .
	poetry run mypy .

# Build local docker containers
build:
	docker-compose build

# Start FastAPI serving API locally
run-api:
	poetry run uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

# Start Celery worker locally
run-worker:
	poetry run celery -A serving.worker worker --loglevel=info -Q heavy,light

# Clean build and pytest cache directories
clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
