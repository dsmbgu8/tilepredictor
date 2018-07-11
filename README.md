# tilepredictor
Platform agnostic (keras currently supported) wrapper for image tile-based CNN classifiers

## generating salience maps using a trained classifier
```
model_flavor=xceptionpartial
model_package=keras
tile_dim=128
tile_bands=3
tile_stride=1
image_dir=/lustre/bbue/ch4/srcfinder/crexp_sub_bilinear
out_dir=$image_dir/salience
load_func=srcfinder_train_util.cmf2rgb_load_func
load_pattern="ang*img_sub_bilinear"
weight_file=/lustre/bbue/ch4/srcfinder/tiles/thompson_thorpe_training/state111417/cmflab_250_4000_tdim${tile_dim}/xceptionpartial_keras/model_iter196_val_loss0.303678_pid58981.h5
CUDA_VISIBLE_DEVICES=0 tilepredictor.py -f $model_flavor -m $model_flavor -w $weight_file --tile_dim $tile_dim --tile_bands $tile_bands --tile_stride $tile_stride --image_dir $image_dir --load_func $load_func --image_load_pattern "$load_pattern" -o $out_dir
```
