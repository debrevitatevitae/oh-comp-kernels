# Makefile for Python project

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_FILES := $(wildcard *.py)
REQUIREMENTS_FILE = requirements.txt
DATA_DIR = data
OUTPUT_DIR = output

# Targets
.PHONY: all install run clean

all: install run

install:
	@$(PIP) install -r $(REQUIREMENTS_FILE)

install-package:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Error: PACKAGE is not set. Usage: make install-package PACKAGE=<package-name>"; \
		exit 1; \
	fi
	@$(PIP) install $(PACKAGE)
	@$(PIP) freeze > $(REQUIREMENTS_FILE)
	@git add $(REQUIREMENTS_FILE)
	@git reset $(filter-out $(REQUIREMENTS_FILE), $(shell git diff --name-only))
	@git commit -m "requirements.txt: update"
	@git add $(filter-out $(REQUIREMENTS_FILE), $(shell git diff --name-only))
	
run:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE is not set. Usage: make run FILE=<file-name>"; \
		exit 1; \
	fi
	@$(PYTHON) $(FILE)

run-all:
	@for file in $(PYTHON_FILES); do \
		$(PYTHON) $$file; \
	done

clean:
	@rm -rf $(OUTPUT_DIR)/*

