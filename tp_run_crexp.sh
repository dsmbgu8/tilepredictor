#!/bin/bash -v

# update the following as necessary
export SRCFINDER_ROOT=/lustre/bbue/ch4/srcfinder
export TP_ROOT_DIR=/lustre/bbue/ch4/tilepredictor
#export TP_EXT_DIR=${HOME}/Research/src/python/external
export TP_EXT_DIR=${TP_ROOT_DIR}/external
# set gpuid='' for cpu, or gpuid="0", "1" or "0,1"" for gpu,
gpuid=''

tp_exe=${TP_ROOT_DIR}/tilepredictor.py
imagedir=${SRCFINDER_ROOT}/crexp_sub_bilinear
outdir=$imagedir/salience

package=keras
flavor=xceptionpartial
ppmm_min=250
ppmm_max=4000
tdim=128
tbands=3
tstride=5

statedir=${SRCFINDER_ROOT}/tiles/thompson_thorpe_training/state111417
modeldir=${statedir}/cmflab_${ppmm_min}_${ppmm_max}_tdim${tdim}
modelweights=model_iter196_val_loss0.303678_pid58981.h5
modelfile=${modeldir}/${flavor}_${package}/${modelweights}

loadfunc=cmf2rgb_load_func.cmf2rgb_load_func_${ppmm_min}_${ppmm_max}


# save CVD value to restore if necessary
CVD_ORIG=${CUDA_VISIBLE_DEVICES}
if [[ "$gpuid" == '' ]]; then
    export CUDA_VISIBLE_DEVICES=''
else
    export CUDA_VISIBLE_DEVICES="$gpuid"
fi

if [ ! -d $outdir ]; then
    mkdir -p $outdir
    echo "created directory \"$outdir\""
fi

# here we go
${tp_exe} -f $flavor -m $package -w $modelfile --tile_dim $tdim \
	  --tile_bands $tbands --tile_stride $tstride \
	  --image_dir $imagedir --output_dir $outdir \
	  --load_func $loadfunc "ang*img_sub_bilinear"

export CUDA_VISIBLE_DEVICES=${CVD_ORIG}
