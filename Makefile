# Makefile for Python project

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_FILES := $(wildcard *.py)
REQUIREMENTS_FILE = requirements.txt
DATA_DIR = data
RESULTS_DIR = results
GRAPHICS_DIR = graphics
BACKUP_DIR = backup
PICKLE_DIR = pickle

# Targets
.PHONY: all backup install run clean

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

# backup everything in results, graphics and pickle directories to backup directory with timestamp in name
backup:
	@mkdir -p $(BACKUP_DIR)
	@tar --exclude '*.gitkeep' -czf $(BACKUP_DIR)/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz $(RESULTS_DIR) $(GRAPHICS_DIR) $(PICKLE_DIR)

# Clean up everything in data/pickle, results and graphics directories
clean:
	@find $(RESULTS_DIR) -type f ! -name '.gitkeep' -delete
	@find $(GRAPHICS_DIR) -type f ! -name '.gitkeep' -delete
	@find $(PICKLE_DIR) -type f ! -name '.gitkeep' -delete
