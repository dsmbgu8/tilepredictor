from warnings import warn
_model_id = 'resnet50'
def model_init(input_shape,**kwargs):
    from keras.applications import ResNet50
    
    assert(len(input_shape)==3)
    
    fix_base = kwargs.pop('fix_base',True)
    
    base_model = ResNet50(weights="imagenet", include_top=False,
                          pooling='avg', input_shape=input_shape)
    if fix_base:
        print('Fixing ResNet50 base_model layers')
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

    return dict(model=base_model,lr_mult=0.1)
