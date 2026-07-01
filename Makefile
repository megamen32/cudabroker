PYTHON ?= python3
VENV ?= .venv
PIP = $(VENV)/bin/pip
PY = $(VENV)/bin/python

.PHONY: venv test run install-unit
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e '.[test]'

test:
	$(PY) -m pytest -q

run:
	$(PY) -m cudabroker.server

install-unit:
	@echo "Install deploy/systemd/cudabroker.service.template manually with USER/WORKDIR/PYTHON substitutions"
