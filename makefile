## makefile automates the build and deployment for python projects

## build
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy
PROJ_ARGS =		-c resources/deepnlp.conf
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache

##doc
PY_DOC_MD_SRC =		./doc/md

#PY_SRC_TEST_PAT ?=	'test_lab*.py'


## targets

include ./zenbuild/main.mk

.PHONY:		exampleclean
exampleclean:
		@for i in example/* ; do \
			make -C $$i clean ; \
		done || true
