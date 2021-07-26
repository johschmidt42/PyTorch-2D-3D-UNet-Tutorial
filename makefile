SHELL=/bin/bash

# MAKEFILE to have in one place all the utility commands
#
# You can use tab for autocomplete on your terminal
# > make[space][tab]
#

check-markdown:
	@./makefile_scripts/check-markdown.sh

check-python:
	@./makefile_scripts/linting-python.sh


