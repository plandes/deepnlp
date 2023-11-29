## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy markdown
PROJ_ARGS =		-c resources/deepnlp.conf
PY_DEP_POST_DEPS +=	modeldeps
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache

## Project
#
EXAMPLE_DIR = 		example


## API docs
#
PY_DOC_MD_SRC =		./doc/md


## Targets
#
include ./zenbuild/main.mk


## Targets
#
# download [spacy models](https://spacy.io/models/en)
.PHONY:		modeldeps
modeldeps:
		$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-model.txt

# clean example output
.PHONY:		exampleclean
exampleclean:
		@for i in $(EXAMPLE_DIR)/* ; do \
			( cd $$i ; ./harness.py clean ) ; \
		done || true

# unit and integration tests
.PHONY:		testall
testall:	test
		@for i in $(EXAMPLE_DIR)/* ; do \
			echo --- testing $$i ; \
			( cd $$i ; ./harness.py clean ; ./harness.py traintest ) ; \
		done

# cleanup noteobook
.PHONY:		resetnb
resetnb:
		@for i in $(EXAMPLE_DIR)/* ; do \
			if [ -d $(EXAMPLE_DIR)/$$i/notebook ] ; then \
				echo resetting $$i ; \
				( cd $(EXAMPLE_DIR)/$$i/notebook ; git checkout . ) ; \
			fi ; \
		done
