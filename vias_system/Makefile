.PHONY: install run test benchmark docs clean

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn vias.main:app --host 0.0.0.0 --port 8000

test:
	python -m unittest discover -s tests -v

benchmark:
	python scripts/benchmark_conflict.py

docs:
	python scripts/build_solution_docx.py

clean:
	rm -rf var var_smoke var_uvicorn
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
