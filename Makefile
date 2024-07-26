#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = revenue-forecasting
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 revenue_forecasting
	isort --check --diff --profile black revenue_forecasting
	black --check --config pyproject.toml revenue_forecasting

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml revenue_forecasting




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	
## Create a ipykernel
.PHONY: create_ipykernel
create_ipykernel: requirements
	$(PYTHON_INTERPRETER) -m pip install ipykernel
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME) (Python $(PYTHON_VERSION))"

	@echo ">>> ipykernel created"

## Create a documentation using numpydoc format
.PHONY: pyment_generate_doc
pyment_generate_doc: 
	pyment -w -o $(DOC_FORMAT) $(PYTHON_FILE)

	@echo ">>> $(DOC_FORMAT) documentation generated"

	
## Update requirements.text
.PHONY: update_requirements
update_requirements: 
	echo '-e .' >requirements.txt
	pip-chill >> requirements.txt

	@echo ">>> requirements.txt updated"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) revenue_forecasting/data/make_dataset.py


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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
