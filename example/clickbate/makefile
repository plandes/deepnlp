PROG =		./run.py
MODEL_DIR =	data/model
MODEL =		glove50


all:		predict

$(MODEL_DIR):	traintest

.PHONY:		traintest
traintest:
		$(PROG) -c models/$(MODEL).conf traintest

.PHONY:		predict
predict:	$(MODEL_DIR)
		cat sample.txt | $(PROG) -c models/$(MODEL).conf predtext

.PHONY:		transpredict
transpredict:
		make MODEL=transformer predict

.PHONY:		help
help:
		$(PROG) --help

.PHONY:		clean
clean:
		find . -type d -name __pycache__ -exec rm -r {} \; || true
		rm -f *.log benchmark.conf

.PHONY:		cleanall
cleanall:	clean
		rm -fr data results

.PHONY:		vaporize
vaporize:	cleanall
		rm -fr corpus model
