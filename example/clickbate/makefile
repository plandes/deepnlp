PROG =		./run.py
MODEL =		data/model


all:		predict

$(MODEL):
		$(PROG) -c models/transformer.conf traintest

.PHONY:		predict
predict:	$(MODEL)
		cat sample.txt | $(PROG) -c models/transformer.conf predtext

.PHONY:		help
help:
		$(PROG) --help

.PHONY:		clean
clean:
		find . -type d -name __pycache__ -exec rm -r {} \; || true
		rm -f *.log

.PHONY:		cleanall
cleanall:	clean
		rm -fr data results

.PHONY:		vaporize
vaporize:	cleanall
		rm -fr corpus model
