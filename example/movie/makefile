## makefile automates the build and deployment for python projects

## build
# project
MTARG =			target
ENV_INC =		$(MTARG)/env.mk
ENTRY =			./run.py
MDL_DIR=		./models
LOG_DIR = 		$(RESULTS_DIR)

## corpora and models
# glove
GLOVE_SRC_URL =		http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
# movie review dataset
STANFORD_SRC_URL =	https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
# cornells labels
CORNELL_SRC_URL = 	https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
# google 300D word2vec
W2V_SRC_URL = 		https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
W2V_DIR =		$(dir $(W2V_PATH))
W2V_FILE =		$(notdir $(W2V_PATH))

include $(ENV_INC)


## targets

$(ENV_INC):
			mkdir -p $(MTARG)
			$(ENTRY) -c $(MDL_DIR)/noop.conf export --expfmt \
				make --expoutput $(ENV_INC)

.PHONY:			info
info:
			@cat $(ENV_INC)

$(GLOVE_DIR):
			@echo "downloading glove embeddings"
			mkdir -p $(GLOVE_DIR)
			curl "$(GLOVE_SRC_URL)" --output $(GLOVE_DIR)/glove.zip
			( cd $(GLOVE_DIR) ; unzip glove.zip )
			rm $(GLOVE_DIR)/glove.zip

$(STANFORD_DIR):
			@echo "downloading the stanford movie review corpus"
			mkdir -p $(STANFORD_DIR)
			curl "$(STANFORD_SRC_URL)" --output $(CORPUS_DIR)/stanford.zip
			( cd $(CORPUS_DIR) ; unzip stanford.zip )
			rm -rf $(CORPUS_DIR)/stanford.zip $(CORPUS_DIR)/__MACOSX

$(CORNELL_DIR):
			@echo "downloading the stanford movie review corpus"
			mkdir -p $(CORNELL_DIR)
			curl "$(CORNELL_SRC_URL)" --output $(CORPUS_DIR)/cornell.tar.gz
			( cd $(CORPUS_DIR) ; tar zxf cornell.tar.gz )
			rm $(CORPUS_DIR)/cornell.tar.gz

$(W2V_PATH):
			@echo "downloading the Google pretrained word vectors"
			mkdir -p $(W2V_DIR)
			curl "$(W2V_SRC_URL)" --output $(W2V_DIR)/$(W2V_FILE).gz
			( cd $(W2V_DIR) ; gunzip $(W2V_FILE).gz )

.PHONY:			corpus
corpus:			$(GLOVE_DIR) $(STANFORD_DIR) $(CORNELL_DIR) $(W2V_PATH)

.PHONY:			stop
stop:
			ps -efa | grep python | grep movie | grep -v grep | \
				awk '{print $$2}' | xargs kill

.PHONY:			cleanstop
cleanstop:
			$(ENTRY) stop

.PHONY:			help
help:
			$(ENTRY) --help

.PHONY:			modelinfo
modelinfo:
			$(ENTRY) info

.PHONY:			rebatch
rebatch:
			rm -rf $(DATA_DIR)/batch
			$(ENTRY) batch -c $(MDL_DIR)/glove50.conf

.PHONY:			modeldebug
modeldebug:
			$(ENTRY) debug --config $(MDL_DIR)/transformer-fixed.conf

.PHONY:			fasttrain
fasttrain:
			$(ENTRY) traintest -p --config $(MDL_DIR)/demo.conf

.PHONY:			predict
predict:
			$(ENTRY) predict --config $(MDL_DIR)/transformer-trainable.conf

.PHONY:			glove50
glove50:
			$(ENTRY) traintest -p --config $(MDL_DIR)/glove50.conf

.PHONY:			glove300
glove300:
			$(ENTRY) traintest -p --config $(MDL_DIR)/glove300.conf

.PHONY:			word2vec
word2vec:
			$(ENTRY) traintest -p --config $(MDL_DIR)/word2vec.conf

.PHONY:			tranfixed
tranfixed:
			$(ENTRY) traintest -p --config $(MDL_DIR)/transformer-fixed.conf

.PHONY:			trantrain
trantrain:
			$(ENTRY) traintest -p --config $(MDL_DIR)/transformer-trainable.conf

.PHONY:			allmodels
allmodels:		glove50 glove300 word2vec transformer

.PHONY:			clean
clean:
			rm -fr $(MTARG) *.log $(RESULTS_DIR)
			find . -type d -name __pycache__ -prune -exec rm -r {} \;

.PHONY:			cleanbatch
cleanbatch:		clean
			rm -fr $(DATA_DIR)/batch

.PHONY:			cleanall
cleanall:		clean
			rm -fr $(DATA_DIR)

.PHONY:			cleancorpus
cleancorpus:
			rm -fr $(CORPUS_DIR)

.PHONY:			vaporize
vaporize:		cleanall cleancorpus
