.PHONY: setup contrast_pairs probes vectors alphasteer test lint

setup:
	pip install -e ".[dev]"

contrast_pairs:
	python scripts/02_generate_contrast_pairs.py

probes:
	python scripts/03_train_probes.py --models all --probes all

vectors:
	python scripts/04_extract_steering_vectors.py --models all --behaviors all

# needs external/AlphaSteer (see README)
alphasteer:
	python scripts/05_train_alphasteer.py --models all

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ scripts/ tests/
