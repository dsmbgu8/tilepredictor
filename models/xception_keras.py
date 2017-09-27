MAX_BLOCK = 14 # xception max block=14
def model_init(input_shape,**kwargs):
    from keras.applications import Xception
    from keras.backend import set_image_data_format
    
    assert(len(input_shape)==3 and input_shape[2]==3)
    set_image_data_format('channels_last')

    fix_base = kwargs.pop('fix_base',True)
    max_block = kwargs.pop('max_block',MAX_BLOCK) 

    base_model = Xception(weights="imagenet", include_top=False,
                          pooling="avg", input_shape=input_shape)
    if fix_base:
        print('Fixing Xceptionv1 base_model layers up to block %d'%max_block)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        trainable = False # fix blocks until we hit max_block
        for i,layer in enumerate(base_model.layers):
            lname = layer.name
            if not trainable and max_block < MAX_BLOCK:
                if lname.startswith('block'):                    
                    spl = lname.split('_')
                    lid = int(spl[0].replace('block',''))
                    if lid>max_block:
                        trainable = True
                        base_model.layers[i-1].trainable = True
                
            layer.trainable = trainable

    return base_model

