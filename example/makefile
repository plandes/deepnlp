# integration tests
.PHONY:		test
test:
		@for i in vectorize clickbate movie ; do \
			echo testing $$i ; \
			make -C $$i testall ; \
		done
