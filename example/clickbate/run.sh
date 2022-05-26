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
$0 <traintest|predict> [glove50|glove300|fasttext|bert]
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
	glove50|'') mname=glove_50;;
	glove300) mname=glove_300;;
	fasttext) mname=fasttext_news_300;;
	bert)
	    conf=models/transformer.conf
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
    prompt "Extremely destruction deletion about to occur, are you sure?"
    if [ "$1" == "--all" ] ; then
	for i in corpus data ; do
	    if [ -d $i ] ; then
		echo "removing $i..."
		rm -r $i
	    fi
	done
    fi
    for i in results data/model ; do
	if [ -d $i ] ; then
	    echo "removing $i..."
	    rm -r $i
	fi
    done
    find . -type d -name __pycache__ -exec rm -fr {} \;
    rm -f *.log benchmark.conf
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
