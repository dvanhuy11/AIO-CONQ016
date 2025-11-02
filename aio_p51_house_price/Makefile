#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = aio_p51_house_price
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create or update conda environment
.PHONY: requirements
requirements:
	@if conda env list | grep -q $(PROJECT_NAME); then \
		echo "Environment $(PROJECT_NAME) exists, updating..."; \
		conda env update --name $(PROJECT_NAME) --file environment.yml --prune; \
	else \
		echo "Environment $(PROJECT_NAME) does not exist, creating..."; \
		conda env create --name $(PROJECT_NAME) -f environment.yml; \
	fi

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
.PHONY: format
format:
	isort src
	black src

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run data preparation script after environment ready
.PHONY: data
data: requirements
	@echo "Activating environment and running dataset script..."
	eval "$$(conda shell.bash hook)"; \
	conda activate $(PROJECT_NAME); \
	$(PYTHON_INTERPRETER) src/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
