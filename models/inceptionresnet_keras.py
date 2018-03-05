from warnings import warn
_model_id='InceptionResNetV2'
def model_init(input_shape,**kwargs):
    from keras.applications import InceptionResNetV2

    assert(len(input_shape)==3)
    
    fix_base = kwargs.pop('fix_base',True)
    if not fix_base:
        warn('%s model fix_base=False, training may take a long time'%_model_id)
    
    base_model = InceptionResNetV2(weights="imagenet", include_top=False,
                                   pooling='avg', input_shape=input_shape)
    if fix_base:
        print('Fixing %s base_model layers'%_model_id)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

    return dict(model=base_model,lr_mult=0.1)

