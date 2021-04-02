## makefile automates the build and deployment for python projects

## build
# project
MTARG =			target
ENV_INC =		$(MTARG)/env.mk
ENTRY_POINT =		./run.py
LOG_DIR = 		.

## corpora and models
CONNL_SRC_URL =		https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data

include $(ENV_INC)


## targets

$(ENV_INC):
			mkdir -p $(MTARG)
			$(ENTRY_POINT) export --output $(ENV_INC)

.PHONY:			info
info:
			@echo "CONNL_DIR: $(CONNL_DIR)"

$(CONNL_DIR):
			@echo "downloading the CONNL 2003 corpus"
			mkdir -p $(CONNL_DIR)
			@for i in test train dev ; do \
				curl "$(CONNL_SRC_URL)/$$i.txt" -L --output $(CONNL_DIR)/$$i.txt ; \
			done

.PHONY:			corpus
corpus:			$(CONNL_DIR)

.PHONY:			env
env:
			cat $(ENV_INC)

.PHONY:			modeltraintest
modeltraintest:
			nohup $(ENTRY_POINT) traintest > traintest.log 2>&1 &

.PHONY:			clean
clean:
			rm -fr $(MTARG) *.log
			find . -type d -name __pycache__ -prune -exec rm -r {} \;

.PHONY:			cleanall
cleanall:		clean
			rm -fr $(MTARG) results data jupyter.log

.PHONY:			cleancorpus
cleancorpus:
			rm -fr $(CORPUS_DIR)

.PHONY:			vaporize
vaporize:		cleanall cleancorpus
			rm -fr $(DATA_DIR)
