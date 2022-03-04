#!/bin/sh

# Re-reun all vectorization, batching, train, test and validation in the
# background as a poor man's integration test.

make cleanall
nohup ./run.py traintest --config models/glove50.conf > test.log 2>&1 &
