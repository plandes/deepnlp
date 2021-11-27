## makefile automates the build and deployment for python projects

## build
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-doc python-doc-deploy
PROJ_ARGS =		-c resources/deepnlp.conf
CLEAN_ALL_DEPS +=	exampleclean
CLEAN_DEPS +=		pycleancache

##doc
PY_DOC_MD_SRC =		./doc/md


## models needed for testing
CORPUS_DIR =		./corpus
GLOVE_DIR =		$(CORPUS_DIR)/glove
# glove
GLOVE_SRC_URL =		http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

PY_SRC_TEST_PAT ?=	'test_trans*.py'


## targets

include ./zenbuild/main.mk

$(GLOVE_DIR):
		@echo "downloading glove embeddings"
		mkdir -p $(GLOVE_DIR)
		curl "$(GLOVE_SRC_URL)" --output $(GLOVE_DIR)/glove.zip
		( cd $(GLOVE_DIR) ; unzip glove.zip )
		rm $(GLOVE_DIR)/glove.zip

.PHONY:		corpus
corpus:		$(GLOVE_DIR)

.PHONY:		exampleclean
exampleclean:
		@for i in example/* ; do \
			make -C $$i clean ; \
		done || true

.PHONY:		cleancorpus
cleancorpus:
		rm -fr $(CORPUS_DIR)
