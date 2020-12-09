## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python
PROJ_MODULES=	git python-doc python-doc-deploy
PROJ_ARGS =	-c resources/deepnlp.conf
CLEAN_DEPS +=	exampleclean

include ./zenbuild/main.mk

.PHONY:		testparse
testparse:
		make PY_SRC_TEST_PAT=test_parse.py test

.PHONY:		testfeatnorm
testfeatnorm:
		make PY_SRC_TEST_PAT=test_featnorm.py test

.PHONY:		testenum
testenum:
		make PY_SRC_TEST_PAT=test_enum.py test

.PHONY:		exampleclean
exampleclean:
		@for i in example/* ; do \
			make -C $$i clean ; \
		done
