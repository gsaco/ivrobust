.PHONY: docs serve figures notebooks links

docs:
	mkdocs build --strict

serve:
	mkdocs serve

figures:
	python scripts/build_figures.py

notebooks:
	pytest --nbmake notebooks/*.ipynb

links:
	python scripts/check_links.py --site-dir docs
