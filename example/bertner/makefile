## makefile automates the build and deployment for python projects

## build
# project
MTARG =			target
ENV_INC =		$(MTARG)/env.mk
ENTRY_POINT =		./run.py
LOG_DIR = 		/d/results

## corpora and models
CONNL_SRC_URL =		https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data

include $(ENV_INC)


## targets

$(ENV_INC):
			mkdir -p $(MTARG)
			$(ENTRY_POINT) export --expfmt make --output $(ENV_INC)

.PHONY:			info
info:
			@cat $(ENV_INC)

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

.PHONY:			rebatch
rebatch:
			rm -rf $(DATA_DIR)/batch
			$(ENTRY_POINT) batch

.PHONY:			traintest
traintest:
			nohup $(ENTRY_POINT) traintest > $(LOG_DIR)/traintest.log 2>&1 &

.PHONY:			stop
stop:
			ps auwx | grep python | grep landes | grep -v grep | \
				grep run.py | awk '{print $$2}' | xargs kill
			ps augwx | grep python | grep landes | grep multipro \
				| grep -v grep | awk '{print $2}' | xargs kill
.PHONY:			clean
clean:
			rm -fr $(MTARG) *.log
			find . -type d -name __pycache__ -prune -exec rm -r {} \;

.PHONY:			cleanall
cleanall:		clean
			rm -fr $(MTARG) results $(DATA_DIR) jupyter.log

.PHONY:			cleancorpus
cleancorpus:
			rm -fr $(CORPUS_DIR)

.PHONY:			vaporize
vaporize:		cleanall cleancorpus
