.PHONY: docs serve figures notebooks links sync-notebooks

sync-notebooks:
	python scripts/sync_docs_notebooks.py

docs: sync-notebooks
	mkdocs build --strict

serve: sync-notebooks
	mkdocs serve

figures:
	python scripts/build_figures.py

notebooks:
	pytest --nbmake notebooks/*.ipynb

links:
	python scripts/check_links.py --site-dir docs
