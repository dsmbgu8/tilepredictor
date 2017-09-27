#!/usr/bin/env bash

flavor=xception #inceptionv3 # cnn3
nbatch=128;
tdim=150;
gpuid=0;
balance="";
predbest="";
dryrun=1;
outdir=$(pwd)/state

if [ $# == 0 ]; then
    echo "usage $0 [-f flavor=$flavor] [-b batchsize=$nbatch] [-d tiledim=$tdim] [-g gpuid=$gpuid] [--balance] [-o outdir=$outdir][-e]"
fi

if [ "$1" == "-f" ]; then
    flavor=$2;
    shift; shift;
fi

if [ "$1" == "-b" ]; then
    nbatch=$2;
    shift; shift;
fi

if [ "$1" == "-d" ]; then
    tdim=$2;
    shift; shift;
fi

if [ "$1" == "-g" ]; then
    gpuid=$2;
    shift; shift;
fi

if [ "$1" == "--balance" ]; then
    balance="--balance"
    shift;
fi

if [ "$1" == "-o" ]; then
    outdir="$2"
    shift; shift;
fi

if [ "$1" == "-e" ]; then
    dryrun=0;
    shift
fi

if [ "$1" == "--pred_best" ]; then
    predbest="--pred_best"
    shift; shift;
fi

nfold=$(ls *_train.txt|wc -l);

echo "gpuid=$gpuid, folds=$nfold"

tilebin=/home/bbue/Research/src/python/util/tilepredictor/tilepredictor_dev.py

seed=42
opts="-f ${flavor} --seed ${seed} -b ${nbatch} -d ${tdim} --save_preds"
opts="$opts ${balance} ${predbest}"
folds="$(seq $nfold)"
for fold in $folds; do
    train_file=$(pwd)/$(ls *${fold}of${nfold}_train.txt)
    test_file=$(pwd)/$(ls *${fold}of${nfold}_test.txt)
    log_file=$(pwd)/fold${fold}of${nfold}_log.txt;
    fout_dir=${outdir}/${fold}of${nfold}/;
    
    args="--train_file ${train_file} --test_file ${test_file}"
    args="${args} -o ${fout_dir} --state_dir ${fout_dir}"
    
    echo "train_file=$train_file"
    echo "test_file=$test_file"
    echo "log_file=$log_file"
    echo "out_dir=$fout_dir"
    if [ $dryrun == 1 ]; then
	echo "CUDA_VISIBLE_DEVICES=${gpuid} python ${tilebin} ${opts} ${args} &> ${log_file}"
	continue
    fi
    CUDA_VISIBLE_DEVICES=${gpuid} python ${tilebin} ${opts} ${args} &> ${log_file}
done
