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
$0 <traintest|predict> [glove50|glove300|fasttext]
$0 stop
$0 clean [--all]"
}

prompt() {
    echo "${1}--CNTRL-C to stop"
    read
}

modelparams() {
    model=$1
    case $1 in
	'') ;;
	glove50|'') mname=glove_50;;
	fasttext) mname=fasttext_news_300;;
	*)
	    echo "$0: unkown model: $1"
	    exit 1
    esac
    if [ ! -z "$mname" ] ; then
	override="--override cb_default.name=$mname"
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
19 Things You'll Only Understand If You're A Woman Who Hates Shaving
Snapchat Ran An Ad For Booze During Its Hajj Story
Poison sue Capitol Records and EMI Music Marketing over royalties
History Of Santa
16 Chocolate Chip Cookies That Prove God Exists
New York City reaches $33 million strip search settlement
21 Movies And Shows To Stream When It's Too Cold Out"
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
