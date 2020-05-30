## makefile automates the build and deployment for python projects

# type of project
PROJ_TYPE=	python
PROJ_MODULES=	git python-doc
PROJ_ARGS =	-c resources/deepnlp.conf

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
