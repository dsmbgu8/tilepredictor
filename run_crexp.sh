#!/bin/bash -v
srcfinderroot=/lustre/bbue/ch4/srcfinder
imagedir=${srcfinderroot}/crexp_sub_bilinear
outdir=$imagedir/salience

package=keras
flavor=xceptionpartial
ppmm_min=250
ppmm_max=4000
tdim=128
tbands=3
tstride=5

statedir=${srcfinderroot}/tiles/thompson_thorpe_training/state111417
modeldir=${statedir}/cmflab_${ppmm_min}_${ppmm_max}_tdim${tdim}
weightfile=model_iter196_val_loss0.303678_pid58981.h5
modelfile=${modeldir}/${flavor}_${package}/${weightfile}

loadfunc=cmf2rgb_load_func.cmf2rgb_load_func_${ppmm_min}_${ppmm_max}

# set CUDA_VISIBLE_DEVICES=0 for gpu, CUDA_VISIBLE_DEVICES='' for cpu
CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} tilepredictor.py -f $flavor \
		    -m $package -w $modelfile --tile_dim $tdim \
		    --tile_bands $tbands --tile_stride $tstride \
		    --image_dir $imagedir --output_dir $outdir \
		    --load_func $loadfunc "ang*img_sub_bilinear"
