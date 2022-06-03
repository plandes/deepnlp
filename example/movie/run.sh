#!/bin/bash

# user specified cation
ACTION=$1
# directory to python virtual environemnt, and binaries
PY_BIN=python
# model entry point
HARNESS=./harness.py

usage() {
    # usage doc
    echo -e "usage:
$0 batch
$0 <traintest|predict> [glove|word2vec|fasttext|bert]
$0 stop
$0 clean [--all]"
}

prompt() {
    echo "${1}--CNTRL-C to stop"
    read
}

batch() {
    echo "rebatching"
    $HARNESS batch --clear
}

modelparams() {
    model=$1
    case $1 in
	'') ;;
	glove) mname=glove_50;;
	word2vec) mname=word2vec_300;;
	fasttext) mname=fasttext_news_300;;
	bert)
	    conf=models/transformer-trainable.conf
	    ;;
	bert-fixed)
	    conf=models/transformer-fixed.conf
	    ;;
	*)
	    echo "unkown model: $1"
	    exit 1
    esac
    if [ ! -z "$conf" ] ; then
	conf="--config $conf"
    fi
    if [ ! -z "$mname" ] ; then
	override="--override mr_default.name=$mname"
    fi
    retval="${conf} ${override}"
}

traintest() {
    modelparams $1
    echo "training and testing using $retval"
    $HARNESS traintest -p $retval
}

stop() {
    touch data/model/update.json
}

predict() {
    modelparams $1
    sample="\
If you sometimes like to go to the movies to have fun , Wasabi is a good place to start.,
There are a few stabs at absurdist comedy ... but mostly the humor is of the sweet , gentle and occasionally cloying kind that has become an Iranian specialty .,
Terrible,
Great movie,
Wonderful, great, awesome, 100%,
Terrible, aweful, worst movie"
    echo "$sample" | $HARNESS predtext - $retval
}

clean() {
    prompt "Destructive deletion about to occur, are you sure?"
    if [ "$1" == "--all" ] ; then
	level=2
    else
	level=1
    fi
    $HARNESS clean --clevel $level
}

case $ACTION in
    batch)
	batch
	;;
    traintest)
	traintest $2
	;;
    predict)
	predict $2
	;;
    stop)
	stop
	;;
    clean)
	clean $2
	;;
    *)
	usage
	exit 1
	;;
esac
