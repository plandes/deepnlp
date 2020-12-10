## makefile automates the build and deployment for python projects

## build
# project
MTARG =			target
ENV_INC =		$(MTARG)/env.mk
MODEL_ARGS =		$(PYTHON_BIN_ARGS) -p -o model_settings.epochs=50
ENTRY_POINT =		PYTHONPATH=../../src/python python ./run.py
LOG_DIR = 		.

## corpora and models
CONNL_SRC_URL =		https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data
# glove
GLOVE_SRC_URL =		http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
# google 300D word2vec
W2V_SRC_URL = 		https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
W2V_DIR =		$(dir $(W2V_PATH))
W2V_FILE =		$(notdir $(W2V_PATH))

include $(ENV_INC)


## targets

$(ENV_INC):
			mkdir -p $(MTARG)
			$(ENTRY_POINT) env > $(ENV_INC)

.PHONY:			info
info:
			@echo "glove-dir: $(GLOVE_DIR)"
			@echo "w2v-dir: $(W2V_DIR)"

$(GLOVE_DIR):
			@echo "downloading glove embeddings"
			mkdir -p $(GLOVE_DIR)
			curl "$(GLOVE_SRC_URL)" --output $(GLOVE_DIR)/glove.zip
			( cd $(GLOVE_DIR) ; unzip glove.zip )
			rm $(GLOVE_DIR)/glove.zip

$(W2V_PATH):
			@echo "downloading the Google pretrained word vectors"
			mkdir -p $(W2V_DIR)
			curl "$(W2V_SRC_URL)" --output $(W2V_DIR)/$(W2V_FILE).gz
			( cd $(W2V_DIR) ; gunzip $(W2V_FILE).gz )

$(CONNL_DIR):
			@echo "downloading the CONNL 2003 corpus"
			mkdir -p $(CONNL_DIR)
			@for i in test train dev ; do \
				curl "$(CONNL_SRC_URL)/$$i.txt" -L --output $(CONNL_DIR)/$$i.txt ; \
			done

.PHONY:			corpus
corpus:			$(CONNL_DIR) $(GLOVE_DIR) $(W2V_PATH)

.PHONY:			env
env:
			cat $(ENV_INC)

.PHONY:			modeltraintest
modeltraintest:
			$(ENTRY_POINT) traintest $(MODEL_ARGS)

.PHONY:			clean
clean:
			rm -fr $(MTARG)
			find . -type d -name __pycache__ -prune -exec rm -r {} \;

.PHONY:			cleanall
cleanall:		clean
			rm -fr $(MTARG) results jupyter.log

.PHONY:			cleancorpus
cleancorpus:
			rm -fr $(CORPUS_DIR)

.PHONY:			vaporize
vaporize:		cleanall cleancorpus
			rm -fr $(DATA_DIR)
