#!/bin/bash

PROG=./run.py
CONF=benchmark.conf

run() {
    model=$1
    batch_proc=$2
    cache=$3

    if [ "$batch_proc" == "y" ] ; then
	bp=True
	echo "removing data dir"
	[ -d data ] && rm -r data
    else
	bp=False
    fi
    if [ "$cache" == "y" ] ; then
	cb=True
	bi=gpu
    else
	cb=False
	bi=buffered
    fi
cat << EOF > $CONF
[time_default]
model=${model}
batch_processed=${bp}
cache_batches=${cb}
batch_iteration=${bi}

[model_settings]
cache_batches = \${time_default:cache_batches}
batch_iteration = \${time_default:batch_iteration}

[dump_model_observer]
add_columns = instance: time_default
file_mode = overwrite
EOF

    ${PROG} -c models/${model}.conf --override $CONF --progress traintest
}

benchmark() {
    echo "removing results dir"
    [ -d results ] && rm -r results
    for model in glove300 transformer-trainable ; do 
	run $model y n # 2
	run $model n n # 0
	run $model y y # 3
	run $model n y # 1
    done
    rm -f $CONF
}

benchmark
