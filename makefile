SHELL=/bin/bash

# MAKEFILE to have in one place all the utility commands
#
# You can use tab for autocomplete in your terminal
# > make[space][tab]
#

#!/usr/bin/env make -f

# Find all PY files except those in hidden folders.
FILES_PY = $(shell find $(CURDIR) -type f -name "*.py" -not -path "$(CURDIR)/.**/**" -not -path "$(CURDIR)/build/**")

format:
	@echo "Running isort..."
	@isort $(FILES_PY)
	@echo "Running black..."
	@black $(FILES_PY)

check-format:
	@echo "Running check isort..."
	@isort --check $(FILES_PY)
	@echo "Running check flake8..."
	@flake8 $(FILES_PY)
