def model_init(input_shape,**kwargs):
    from keras.applications import InceptionV3
    from keras.backend import set_image_data_format
    
    assert(len(input_shape)==3 and input_shape[2]==3)
    set_image_data_format('channels_last')
    
    fix_base = kwargs.pop('fix_base',True)
    
    base_model = InceptionV3(weights="imagenet", include_top=False,
                             pooling='avg', input_shape=input_shape)
    if fix_base:
        print('Fixing InceptionV3 base_model layers')
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

    return base_model

