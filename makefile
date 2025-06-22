## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/deploy
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache


## Project
#
EXAMPLE_DIR = 		example


## Targets
#
include ./zenbuild/main.mk


## Targets
#
# clean example output
.PHONY:			exampleclean
exampleclean:
			@for i in $(EXAMPLE_DIR)/* ; do \
				( cd $$i ; ./harness.py clean ) ; \
			done || true

# unit and integration tests
.PHONY:			testall
testall:		test
			@for i in $(EXAMPLE_DIR)/* ; do \
				echo --- testing $$i ; \
				( cd $$i ; ./harness.py clean ; ./harness.py traintest ) ; \
			done

# cleanup noteobook
.PHONY:			resetnb
resetnb:
			@for i in $(EXAMPLE_DIR)/* ; do \
				if [ -d $(EXAMPLE_DIR)/$$i/notebook ] ; then \
					echo resetting $$i ; \
					( cd $(EXAMPLE_DIR)/$$i/notebook ; git checkout . ) ; \
				fi ; \
			done
