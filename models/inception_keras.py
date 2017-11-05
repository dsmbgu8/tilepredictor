from warnings import warn
_model_id = 'InceptionV3'
def model_init(input_shape,**kwargs):
    from keras.applications import InceptionV3
    from keras.backend import set_image_data_format
    
    assert(len(input_shape)==3 and input_shape[2]==3)
    set_image_data_format('channels_last')

    fix_base = kwargs.pop('fix_base',True)
    if not fix_base:
        warn('%s model fix_base=False, training may take a long time'%_model_id)
    
    base_model = InceptionV3(weights="imagenet", include_top=False,
                             pooling='avg', input_shape=input_shape)
    if fix_base:
        print('Fixing %s base_model layers'%_model_id)
        # first: train only the top layers (which were randomly initialized)
        
        for layer in base_model.layers:
            layer.trainable = False

    return base_model

