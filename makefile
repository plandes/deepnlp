## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_TEST_ALL_TARGETS +=	testexamples
CLEAN_ALL_DEPS +=	cleanexamples


## Project
#
EXAMPLE_DIR = 		example


## Targets
#
include ./zenbuild/main.mk


## Targets
#
# clean example output
.PHONY:			cleanexamples
cleanexamples:
			@export PYTHONPATH="${HOME}/view/reslib/deepnlp" ; \
			 $(MAKE) $(PY_MAKE_ARGS) pytestrun \
				ARG="for i in $(EXAMPLE_DIR)/* ; \
					do ( cd \$$\$$i ; ./harness.py clean ) ; \
				done"

# unit and integration tests
.PHONY:			testexamples
testexamples:
			@export PYTHONPATH="${HOME}/view/reslib/deepnlp" ; \
			 $(MAKE) $(PY_MAKE_ARGS) pytestrun \
				ARG="for i in $(EXAMPLE_DIR)/* ; \
					do ( cd \$$\$$i ; ./harness.py clean ; \
					     ./harness.py traintest ) ; \
				done"
