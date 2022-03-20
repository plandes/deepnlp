## makefile automates the build and deployment for python projects

## build
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy
PROJ_ARGS =		-c resources/deepnlp.conf
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache

## project
EXAMPLE_DIR = 		example
EXAMPLE_NAMES = 	vectorize clickbate movie ner 

## doc
PY_DOC_MD_SRC =		./doc/md

#PY_SRC_TEST_PAT ?=	'test_lab*.py'


## targets

include ./zenbuild/main.mk

.PHONY:		exampleclean
exampleclean:
		@for i in $(EXAMPLE_DIR)/* ; do \
			make -C $$i clean ; \
		done || true

.PHONY:		testall
testall:	test
		@for i in $(EXAMPLE_NAMES) ; do \
			echo testing $$i ; \
			make -C $(EXAMPLE_DIR)/$$i testall ; \
		done

.PHONY:		restnb
resetnb:
		@for i in $(EXAMPLE_NAMES) ; do \
			if [ -d $(EXAMPLE_DIR)/$$i/notebook ] ; then \
				echo resetting $$i ; \
				( cd $(EXAMPLE_DIR)/$$i/notebook ; git checkout . ) ; \
			fi ; \
		done
