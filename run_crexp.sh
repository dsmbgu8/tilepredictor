#!/bin/bash -v
srcfinderroot=/lustre/bbue/ch4/srcfinder
imagedir=${srcfinderroot}/crexp_sub_bilinear
outdir=$imagedir/salience

ppmm_min=250
ppmm_max=4000
tile_dim=128
tile_bands=3
tile_stride=1

statedir=${srcfinderroot}/tiles/thompson_thorpe_training/state111417
modeldir=${statedir}/cmflab_${ppmm_min}_${ppmm_max}_tdim${tile_dim}
model_flavor=xceptionpartial
model_package=keras
model_weights=model_iter196_val_loss0.303678_pid58981.h5
model_file=${modeldir}/${model_flavor}_${model_package}/${model_weights}

load_func=cmf2rgb_load_func.cmf2rgb_load_func_${ppmm_min}_${ppmm_max}
load_pattern="ang*img_sub_bilinear"
# set CUDA_VISIBLE_DEVICES=0 for gpu, CUDA_VISIBLE_DEVICES='' for cpu
CUDA_VISIBLE_DEVICES=''
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} tilepredictor.py -f $model_flavor \
		    -m $model_package -w $model_file --tile_dim $tile_dim \
		    --tile_bands $tile_bands --tile_stride $tile_stride \
		    --image_dir $imagedir --load_func $load_func \
		    --output_dir $outdir --image_load_pattern "$load_pattern"
