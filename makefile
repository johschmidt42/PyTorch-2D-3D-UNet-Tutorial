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

clean:
	@find . -name '__pycache__' -exec rm -fr {} +
	@find . -name '.pytest_cache' -exec rm -fr {} +
	@find . -name '.coverage' -exec rm -f {} +
	@find . -name 'coverage.xml' -exec rm -f {} +
	@find . -name 'pytest-results.xml' -exec rm -f {} +
	@find . -name 'build' -exec rm -fr {} +
	@find . -name 'dist' -exec rm -fr {} +
	@find . -name '*.egg-info' -exec rm -Rf {} +
