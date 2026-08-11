.PHONY: install run test demo

VENV := .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,demo]"

run:
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

test:
	$(VENV)/bin/pytest -q

demo:
	$(VENV)/bin/streamlit run demo/streamlit_app.py
