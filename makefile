## makefile automates the build and deployment for python projects

## build
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy
PROJ_ARGS =		-c resources/deepnlp.conf
PY_DEP_POST_DEPS +=	modeldeps
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache

## project
EXAMPLE_DIR = 		example

## doc
PY_DOC_MD_SRC =		./doc/md

#PY_SRC_TEST_PAT ?=	'test_lab*.py'


## targets

include ./zenbuild/main.mk

# https://spacy.io/models/en
.PHONY:		allmodels
allmodels:
		@for i in $(SPACY_MODELS) ; do \
			echo "installing $$i" ; \
			$(PYTHON_BIN) -m spacy download en_core_web_$${i} ; \
		done

.PHONY:		modeldeps
modeldeps:
		$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-model.txt

.PHONY:		exampleclean
exampleclean:
		@for i in $(EXAMPLE_DIR)/* ; do \
			( cd $$i ; ./harness.py clean ) ; \
		done || true

.PHONY:		testall
testall:	test
		@for i in $(EXAMPLE_DIR)/* ; do \
			echo testing $$i ; \
			( cd $$i ; ./harness.py clean ; ./harness.py traintest ) ; \
		done

.PHONY:		resetnb
resetnb:
		@for i in $(EXAMPLE_DIR)/* ; do \
			if [ -d $(EXAMPLE_DIR)/$$i/notebook ] ; then \
				echo resetting $$i ; \
				( cd $(EXAMPLE_DIR)/$$i/notebook ; git checkout . ) ; \
			fi ; \
		done
