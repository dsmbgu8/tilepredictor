from warnings import warn
_model_id = 'VGG16'
def model_init(input_shape,**kwargs):
    from keras.applications import VGG16
    
    assert(len(input_shape)==3)

    fix_base = kwargs.pop('fix_base',False)
    if not fix_base:
        warn('%s model fix_base=False, training may take a long time'%_model_id)

    base_model = VGG16(input_shape=input_shape, pooling='avg',
                       weights="imagenet", include_top=False)
                       
    if fix_base:
        print('Fixing %s base_model layers'%_model_id)
        # first: train only the top layers (which were randomly initialized)
        for layer in base_model.layers:
            layer.trainable = False

    return dict(model=base_model,lr_mult=0.5)
