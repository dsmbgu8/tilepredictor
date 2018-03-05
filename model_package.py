from __future__ import absolute_import, division, print_function
import sys
import time
gettime = time.time

from pylib import *

pyext=expanduser('~/Research/src/python/external')
#sys.path.insert(0,pathjoin(pyext,'keras2/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras204/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras207/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras208/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras-multiprocess-image-data-generator'))
sys.path.insert(0,pathjoin(pyext,'CLR')) # cyclic learning rate callback
sys.path.insert(0,pathjoin(pyext,'imgaug')) # image augmentation

from tilepredictor_util import *

valid_packages = ['keras']
model_path = pathjoin(tilepredictor_home,'models')
valid_flavors = []
for pkg in valid_packages:
    for path in glob(pathjoin(model_path,'*'+pkg+'*.py')):
        model_dir,flavor_file = pathsplit(path)
        valid_flavors.append(flavor_file.split('_')[0])
                          
valid_flavors = list(set(valid_flavors))

default_package   = valid_packages[0] # 'keras'
default_flavor    = valid_flavors[0] # 'cnn3'
default_state_dir = pathjoin(os.getcwd(),'state')

datagen_paramf    = pathjoin(tilepredictor_home,'datagen_params.json')

# softmax probs are:
# [0.0,0.5] class 0
# [0.5,1.0] class 1
# by default, we map these to [0.0,0.5] and store the class label separately
# scale_probs maps the [0.0,0.5] probs to [0.0,1.0] per class
scale_probs = True
randomize = False

# network architecture / training parameters
default_n_hidden = 1024 # nodes in last FC layer before classification
default_n_classes = 2 # we currently only consider binary classification problems 

# optimizer parameters
batch_size = 32
n_epochs = 2000
n_batches = n_epochs//batch_size


use_multiprocessing = False
n_workers = 4

random_state = 42
tol = 0.000001
optparams = dict(
    weight_decay = 0.000001,
    reduce_lr = 0.1,
    beta_1 = 0.9,
    beta_2 = 0.999,
    lr_min = 0.0001,
    lr_max = 0.01,
    obj_lambda2 = 0.0025,
    max_norm = np.inf, # 5.0
    tol = tol,
    stop_delta = 0.001
)


def load_datagen_params(paramf,verbose=0):
    if verbose:
        print('Loading datagen parameters from "%s"'%paramf)
    return load_json(paramf)

def save_datagen_params(paramf,params,verbose=0,**kwargs):
    if verbose:
        print('Saving datagen parameters to "%s"'%paramf)    
    return save_json(paramf,params,**kwargs)
        
@threadsafe_generator
def datagen_arrays(X,y,batch_size,datagen_params,shuffle=True,
                   fill_partial=False,random_state=random_state,
                   preprocessing_function=None,verbose=0):

    datagen_model = datagen_params.pop('model','ImageDataGenerator')
    if datagen_model == 'ImageDataGenerator':
        from keras.preprocessing.image import ImageDataGenerator
        # only call datagen.fit() if these keys are present
        fit_kw = ['featurewise_center',
                  'featurewise_std_normalization',
                  'zca_whitening']
        flowkw = dict(batch_size=batch_size,shuffle=shuffle,seed=random_state,
                      save_to_dir=None,save_prefix='',save_format='png')
        datagen_params.setdefault('preprocessing_function',preprocessing_function)
        datagen = ImageDataGenerator(**datagen_params)
        # only fit datagen if one of the fit keys is true
        if any([datagen_params.get(key,False) for key in fit_kw]):
            print('Fitting ImageDataGenerator for %d samples (this could take some time)'%len(X))
            fittime = gettime()
            datagen.fit(X,seed=random_state)
            fittime = gettime()-fittime
            print('Fit complete, processing time: %0.3f seconds'%fittime)
        datagen_transform = datagen.random_transform
        datagen_iter = datagen.flow(X,y,**flowkw)
    elif datagen_model == 'imgaug':
        raise Exception('imgaug datagen not implemented yet')
    else:
        raise Exception('unknown datagen_model "%s"'%datagen_model)
    for bi, (X_batch, y_batch) in enumerate(datagen_iter):
        if is_collection(X_batch):
            warn('ImageDataGenerator generates ImageCollections, concatenating')
            X_batch = X_batch.concatenate()

        if fill_partial and X_batch.ndim==4 and X_batch.shape[0] < batch_size:
            X_fill,y_fill = fill_batch(X_batch,y_batch,batch_size,balance=True)
            X_batch = np.r_[X_batch,map(datagen_transform,X_fill)]
            y_batch = np.r_[y_batch,y_fill]
        if verbose>1 and bi==0:
            print('\nBatch %d: '%bi,
                  'X_batch.shape: %s,'%str(X_batch.shape),
                  'y_batch.shape: %s'%str(y_batch.shape))
            band_stats(X_batch,verbose=1)
            class_stats(to_binary(y_batch),verbose=1)

        yield X_batch, y_batch

@threadsafe_generator
def datagen_directory(path,target_size,batch_size,datagen_params,
                      classes=None,class_mode='categorical',
                      shuffle=True,fill_partial=False,verbose=0,
                      preprocessing_function=None, random_state=random_state):
    from keras.preprocessing.image import ImageDataGenerator

    # only call datagen.fit() if these keys are present
    fit_kw = ['zca_whitening','featurewise_center',
              'featurewise_std_normalization']
    flowkw = dict(class_mode=class_mode, classes=classes, color_mode='rgb', 
                  batch_size=batch_size, target_size=target_size,
                  shuffle=shuffle, seed=random_state, follow_links=True,
                  save_to_dir=None, save_prefix='', save_format='png')
    datagen_params.setdefault('preprocessing_function',preprocessing_function)
    datagen = ImageDataGenerator(**datagen_params)
    # only fit datagen if one of the fit keys is true
    if any([datagen_params.get(key,False) for key in fit_kw]):
        print('Fitting ImageDataGenerator for %d samples (this could take awhile)'%len(X))
        fittime = gettime()
        datagen.fit(X,seed=random_state)
        print('Fit complete, processing time: %0.3f seconds'%gettime()-fittime)
    transform = datagen.random_transform
    datagen_iter = datagen.flow_from_directory(path,**flowkw)
    for bi, (X_batch, y_batch) in enumerate(datagen_iter):
        bi_collection = is_collection(X_batch)
        if bi_collection:
            X_batch = X_batch.concatenate()
        
        if fill_partial and X_batch.ndim==4 and X_batch.shape[0] < batch_size:
            # fill a partial batch with balanced+transformed inputs
            X_fill,y_fill = fill_batch(X_batch,y_batch,batch_size,
                                       balance=True)
            if X_fill.ndim == X_batch.ndim-1: # filling with a single sample
                X_fill = X_fill[np.newaxis]
                y_fill = y_fill[np.newaxis]
            
            X_batch = np.r_[X_batch,map(transform,X_fill)]
            y_batch = np.r_[y_batch,y_fill]
        if verbose>1 and bi==0:
            print('\n\nBatch %d: '%bi,
                  'is_collection: %s,'%str(bi_collection),
                  'X_batch.shape: %s,'%str(X_batch.shape),
                  'y_batch.shape: %s'%str(y_batch.shape))
            band_stats(X_batch,verbose=1)
            class_stats(to_binary(y_batch),verbose=1)
        yield X_batch, y_batch

def parse_model_meta(modelf,val_monitor='val_loss'):
    """
    parse_meta(modelf)

    Summary: parses initial_epoch and initial_monitor from a model or weight filename

    Arguments:
    - self: self
    - modelf: filename containing epoch/monitor values 

    Keyword Arguments:
    None

    Output:
    - initial_epoch
    - initial_monitor

    """

    start_epoch = 0
    start_monitor = None
    val_type = val_monitor.replace('val_','')
    fields = basename(modelf).split('_')
    msg = []
    for field in fields:
        for epoch_key in ('iter','epoch'):
            if field.startswith(epoch_key):
                start_epoch = int(field.replace(epoch_key,''))
                msg.append('start_epoch value=%d'%start_epoch)
        for monitor_key in ('loss','fscore'):
            if monitor_key in field:
                if monitor_key != val_type:
                    warn('found monitor "%s" that differs from model.val_type="%s", ignored')
                    continue
                start_monitor = float(field.replace(monitor_key,''))
                msg.append('start_monitor value=%.6f'%start_monitor)

    if len(msg)!=0:
        print('Parsed',', '.join(msg),'from',modelf)
                
    return start_epoch, start_monitor
    
class Model(object):
    """
    Model: wrapper class for package model predictor functions
    """
    def __init__(self, **kwargs):
        self.initialized   = False        
        self.flavor        = kwargs['flavor']
        self.package       = kwargs['package']
        self.backend       = kwargs['backend']
        self.base          = kwargs['base']
        self.params        = kwargs['params']
        self.input_shape   = kwargs['input_shape']
        self.batch         = kwargs['batch']
        self.predict       = kwargs['predict']
        self.transform     = kwargs['transform']
        self.state_dir     = kwargs['state_dir']
        self.load_base     = kwargs['load_base']

        self.transpose     = kwargs['transpose']
        self.rtranspose    = kwargs['rtranspose']
        
        self.pid           = kwargs['pid']
        self.start_epoch   = kwargs['start_epoch']
        self.start_monitor = kwargs['start_monitor']
        self.val_monitor   = kwargs['val_monitor']

        self.callbacks     = []
        
        self.val_type      = self.val_monitor.replace('val_','')
        self.val_best      = None
        self.val_cb        = None
        
        model_suf = '_'.join([self.flavor,self.package])
        self.model_dir = pathjoin(self.state_dir,model_suf)
        if not pathexists(self.model_dir):
            makedirs(self.model_dir,verbose=True)

    def preprocess(self,img,transpose=True,verbose=0):
        shape = img.shape
        n_bands = shape[-1]
        dtype = img.dtype
        
        if img.ndim not in (3,4) or n_bands not in (1,3):
            warn('No preprocessing function defined for image shape "%s"!'%str(shape))

        if dtype == np.uint8:
            #print('Preprocessing function: preprocess_img_u8')
            _preprocess = preprocess_img_u8
        elif dtype in (np.float32,np.float64):
            #print('Preprocessing function: preprocess_img_float')
            _preprocess = preprocess_img_float
        else:
            raise Exception('No preprocessing function defined for data type "%s"!'%str(dtype))

        imgpre = _preprocess(img)
        
        if verbose:
            if img.ndim == 3:
                imin,imax,_ = band_stats(img[np.newaxis])
                omin,omax,_ = band_stats(imgpre[np.newaxis])
            else:
                imin,imax,_ = band_stats(img)
                omin,omax,_ = band_stats(imgpre)
                
        if transpose:
            if imgpre.ndim==3:
                imgpre = imgpre.transpose(self.transpose) 
            elif imgpre.ndim==4:
                imgpre = imgpre.transpose([0]+[i+1 for i in self.transpose])
                
        if verbose:
            print('Before preprocess: '
                  'type=%s, shape=%s, '%(str(dtype),str(shape)),
                  'range = %s'%str(map(list,np.c_[imin,imax])))
            otype = imgpre.dtype
            oshape = imgpre.shape
            print('After preprocess: '
                  'type=%s, shape=%s, '%(str(otype),str(oshape)),
                  'range = %s'%str(map(list,np.c_[omin,omax])))
        
        return imgpre

    def compile(self):
        self.base.compile(**self.params)
    
    def save(self,modelf,**kwargs):
        basef,ext = splitext(modelf)
        weightf = basef+'.h5'
        classf = basef+'.json'
        kwargs.setdefault('overwrite',True)
        save_json(classf,self.__dict__)
        self.base.save(weightf,**kwargs)
    
    def load(self,modelf,**kwargs):
        self.start_epoch, self.start_monitor = parse_model_meta(modelf)
        self.base = self.load_base(modelf,**kwargs)
            
    def save_weights(self, weightf, **kwargs):
        kwargs.setdefault('overwrite',True)
        self.base.save_weights(weightf,**kwargs)
    
    def load_weights(self, weightf, **kwargs):
        self.start_epoch, self.start_monitor = parse_model_meta(weightf)        
        self.base.load_weights(weightf,**kwargs)
        self.initialized = True

    def init_callbacks(self,n_epochs=n_epochs,n_batches=n_batches,**kwargs):
        from keras.callbacks import TerminateOnNaN, EarlyStopping, \
            ReduceLROnPlateau, CSVLogger, TensorBoard
        from validation_checkpoint import ValidationCheckpoint
        from clr_callback import CyclicLR

        print('Initializing model callbacks')
        use_tensorboard = kwargs.pop('use_tensorboard',False)
        val_monitor = kwargs.pop('monitor','val_loss')
        stop_early = kwargs.pop('stop_early',int(n_epochs*0.2))
        step_lr = kwargs.pop('step_lr',None)
        clr_loop = kwargs.pop('clr_loop',None)

        # strides to test/save model during training        
        test_epoch = kwargs.pop('test_epoch',1)
        save_epoch = kwargs.pop('save_epoch',1)
        random_state = kwargs.pop('random_state',42)
        verbose = kwargs.pop('verbose',1)        
        # exit early if the last [stop_early] test scores are all worse than the best

        save_preds = kwargs.pop('save_preds',True)    
        save_model = kwargs.pop('save_model',True) 
        
        model_dir = self.model_dir
        initial_monitor = self.start_monitor
        initial_epoch = self.start_epoch
        
        # configure callbacks
        val_mode = 'auto'

        ctimestr = epoch2str(gettime())
    
        train_logf = pathjoin(model_dir,'training_log_%s_pid%d.csv'%(ctimestr,
                                                                     self.pid))
        # if pathexists(train_logf) and pathsize(train_logf) != 0:
        #     #ctimestr = epoch2str(gettime())
        #     #ctimestr = epoch2str(pathctime(train_logf))
        #     ctimestr = '1'
        #     logf_base,logf_ext = splitext(train_logf)
        #     old_logf = logf_base+'_'+ctimestr+logf_ext
        #     print('Backing up existing log file "%s" to "%s"'%(train_logf,old_logf))
        #     os.rename(train_logf,old_logf)

        self.val_monitor = val_monitor
        self.save_preds = save_preds
        self.save_model = save_model
        self.val_period = save_epoch
            
        self.val_cb = ValidationCheckpoint(val_monitor=val_monitor,
                                           save_best_preds=save_preds,
                                           save_best_model=save_model,
                                           model_dir=model_dir,
                                           mode=val_mode,pid=self.pid,
                                           initial_monitor=initial_monitor,
                                           initial_epoch=initial_epoch,
                                           period=save_epoch, verbose=verbose)
        #self.val_cb = ModelCheckpoint(model_iterf,monitor=val_monitor,mode=val_mode, period=save_epoch,
        #                        save_best_only=True, save_weights_only=False,                                
        #                        verbose=False)
        step_lr = step_lr or int(n_batches*4)
        self.lr_cb = CyclicLR(base_lr=optparams['lr_min'],
                              max_lr=optparams['lr_max'],
                              step_size=step_lr,
                              loop=clr_loop)
        # else:
        #     step_lr = step_lr or min(100,int(n_epochs*0.01))
        #     self.lr_cb = ReduceLROnPlateau(monitor=val_monitor,
        #                                    mode=val_mode,
        #                                    patience=step_lr,
        #                                    min_lr=optparams['lr_min'],
        #                                    factor=optparams['reduce_lr'],
        #                                    epsilon=optparams['tol'],
        #                                    verbose=verbose)
        
        self.es_cb = EarlyStopping(monitor=val_monitor, patience=stop_early,
                                   min_delta=optparams['stop_delta'],
                                   mode=val_mode, verbose=verbose)
        self.tn_cb = TerminateOnNaN()

        
        self.cv_cb = CSVLogger(filename=train_logf,append=True)
        self.callbacks = [self.val_cb,self.lr_cb,self.es_cb,
                          self.tn_cb,self.cv_cb]

        if self.backend=='tensorflow' and use_tensorboard:
            tb_batch_size=32
            tb_histogram_freq = 1
            tb_embeddings_freq = 0
            tb_log_dir = pathjoin(model_dir,'tb_logs_pid%d'%self.pid)
            if not pathexists(tb_log_dir):
                os.makedirs(tb_log_dir)
            
            self.tb_cb = TensorBoard(log_dir=tb_log_dir,
                                     histogram_freq=tb_histogram_freq,
                                     batch_size=tb_batch_size,
                                     write_graph=True,
                                     write_grads=True,
                                     write_images=True,
                                     embeddings_freq=tb_embeddings_freq,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)
            self.callbacks.append(self.tb_cb)
        elif self.backend!='tensorflow' and use_tensorboard:
            print('Cannot use tensorboard with backend "%s"'%self.backend)
            use_tensorboard=False

        print('Initialized %d callbacks:'%len(self.callbacks),
              str(self.callbacks))
    
    def write_mispreds(self, outf, mispred_ids):
        n_mispred=len(mispred_ids)
        if n_mispred==0:
            return
        with open(outf,'w') as fid:
            print('# %d mispreds'%n_mispred,file=fid)
            for i,m_id in enumerate(mispred_ids):
                print('%s'%(str(m_id)),file=fid)
                
    def train(self,train_gen,n_epochs,n_batches,initial_epoch=None,
              validation_data=None,validation_ids=[],validation_preds=[],
              **kwargs):

        verbose = kwargs.get('verbose',1)
        initial_epoch = initial_epoch or self.start_epoch
        total_epochs = kwargs.pop('total_epochs',n_epochs-initial_epoch)
        if len(self.callbacks)==0:
            # specify total_epochs separately in case we're only training
            # for a subset of the total number of epochs
            self.init_callbacks(n_epochs=total_epochs,
                                n_batches=n_batches,**kwargs)

        if len(validation_ids)!=0 and self.val_cb:
            # always update ids in case validation_data changed
            self.val_cb.update_ids(validation_ids)

        print(', '.join(['Training network for %d epochs'%n_epochs,
                         'starting at epoch %d'%initial_epoch,
                         '%d batches/epoch'%n_batches])) 
        self.base.fit_generator(train_gen,n_batches,
                                epochs=n_epochs,
                                initial_epoch=initial_epoch,
                                validation_data=validation_data,
                                callbacks=self.callbacks,
                                workers=n_workers,
                                use_multiprocessing=use_multiprocessing,
                                verbose=1)

        if len(validation_preds)!=0 and self.val_cb:
            # always update ids in case validation_data changed
            validation_preds[:] = self.val_cb.collect_predictions()

        self.initialized = True

def model_summary(model):
    layers = model.layers
    l = layers[0]
    lclass = l.__class__.__name__
    print('Model: %d layers'%len(layers))
    print('Layer[0] (%s) input_shape: %s'%(lclass,str(l.input_shape)))
    prev_shape = l.output_shape
    for i,l in enumerate(layers):
        if l.output_shape == prev_shape:
            continue
        lclass = l.__class__.__name__
        print('Layer[%d] (%s) output_shape: %s'%(i,lclass,str(l.output_shape)))

        
def compile_model(input_shape,n_classes,n_bands,n_hidden=None,**kwargs):
    from keras.backend import image_data_format,set_image_data_format
    import importlib

    package   = kwargs.pop('model_package',default_package)
    flavor    = kwargs.pop('model_flavor',default_flavor)
    state_dir = kwargs.pop('model_state_dir',None)
    weightf   = kwargs.pop('model_weightf',None)
    flavorp   = kwargs.pop('flavor_params',{})
    num_gpus  = kwargs.pop('num_gpus',0)

    
    # new paths: e.g., state_dir/cnn3_keras
    state_suf = '_'.join([flavor,package])
    if pathexists(pathjoin(state_dir,package,flavor)):
        # old paths: e.g., state_dir/keras/cnn3
        state_suf = pathjoin(package,flavor)
        
    if weightf and not pathexists(weightf):
        # first check if weight file exists in state_dir
        if state_dir and pathexists(pathjoin(state_dir,state_suf,weightf)):
            print('Found weight file "%s" in state_dir "%s"'%(weightf,state_dir))
            weightf = pathjoin(state_dir,state_suf,weightf)
        else:
            # otherwise bail out due to the bad path
            raise Exception('Weight file "%s" not found'%weightf)
            

    # need to specify hidden layer if we don't inherit from weight file
    if not weightf:
        n_hidden = n_hidden or default_n_hidden 

    output_shape = [n_hidden,n_classes]
    
    if not state_dir:
        if weightf:
            model_dir,weight_file = pathsplit(weightf)
            state_dir = model_dir.replace(state_suf,'')
            state_dir = state_dir.replace('//','/')
            print('Using model state_dir="%s"'%state_dir)
        else:
            state_dir = default_state_dir
    model_dir = pathjoin(state_dir,state_suf)

    if package not in valid_packages:
        packages_str = str(valid_packages)
        warn('Invalid model_package: "%s" (packages=%s)'%(package,packages_str))
        sys.exit(1)

    if flavor not in valid_flavors:
        flavors_str = str(valid_flavors)
        warn('Invalid model_flavor: "%s" (flavors=%s)'%(flavor,flavors_str))
        sys.exit(1)

    package_id  = "{package}_model".format(**locals())
    flavor_id   = "models.{flavor}_{package}".format(**locals())    
    package_lib = importlib.import_module(package_id)    
    flavor_lib  = importlib.import_module(flavor_id)
    model_backend = package_lib.backend().backend()
    model_pid     = os.getpid()
    
    backend_image_format = kwargs.pop('backend_image_format',True)

    # default image__format = tensorflow = channels_last 
    model_transpose = [0,1,2]
    model_rtranspose = [0,1,2]
    image_format = 'channels_last'
    if backend_image_format:
        if model_backend=='tensorflow':
            image_format = 'channels_last'
            if input_shape[0]==n_bands:
                warn('Converted input_shape "%s" to "channels_last"'
                     ' format for tensorflow backend')
                input_shape = input_shape[1:]+[3]
                model_transpose = [2,0,1]
                model_rtranspose = [1,2,0]
        elif model_backend=='theano':
            image_format = 'channels_first'
            if input_shape[-1]==n_bands:
                warn('Converted input_shape "%s" to "channels_first"'
                     ' format for theano backend')        
                input_shape = [3]+input_shape[:-1]
                model_transpose = [2,0,1]
                model_rtranspose = [1,2,0]
    set_image_data_format(image_format)

    print('Initializing new %s_%s model with:'%(flavor,package),
          '\ninput_shape=%s,'%str(input_shape),
          'output_shape=%s,'%str(output_shape),
          'model_transpose=%s,'%str(model_transpose),
          'image_data_format=%s'%image_data_format())
                
    model_params = flavor_lib.model_init(input_shape,**flavorp)
    lr_mult = model_params.pop('lr_mult',1.0)        

    if weightf:
        start_epoch, start_monitor = parse_model_meta(weightf)
        print('Restoring existing %s_%s model'%(flavor,package),
              'from file: "%s" with:'%weightf,
              '\ninput_shape=%s,'%str(input_shape),
              'output_shape=%s,'%str(output_shape))
        model_base = package_lib.load_model(weightf)
    else:        
        model_base = model_params['model']
        start_epoch = 0
        start_monitor = None
        model_base = package_lib.update_base_outputs(model_base,output_shape,
                                                     optparam=optparams)
    model_input_shape = model_base.layers[0].input_shape
    model_output_shape = model_base.layers[-1].output_shape
    
    assert(tuple(input_shape) == model_input_shape[-len(input_shape):])
    assert(output_shape[-1] == model_output_shape[-1])    
    model_params['model'] = model_base
    print('Initialized model with:',
          '\nmodel_input_shape=%s,'%str(model_input_shape),
          'model_output_shape=%s,'%str(model_output_shape),
          'start_epoch=%d,'%start_epoch,
          'start_monitor=',start_monitor)
    
    if lr_mult!=1.0:
        lr_upkeys = []
        for key,val in optparams.iteritems():
            if key.startswith('lr_'):
                optparams[key] = optparams[key]*lr_mult
                lr_upkeys.append(key)
                
        print('Updated optparams "%s" with lr_mult=%6.3f'%(str(lr_upkeys),
                                                           lr_mult))

    if model_backend == 'tensorflow':
        from keras.utils.training_utils import multi_gpu_model
        from tensorflow import device as tfdevice
        if num_gpus==1:
            print('Building single-GPU tensorflow model')
            # build and run on single GPU
            with tfdevice("/gpu:0"):
                model_base.build()
        elif num_gpus > 1:
            print('Building multi-GPU tensorflow model')
            # build on the CPU, distribute to multiple GPUs
            with tfdevice("/cpu:0"):
                model_base.build()
            
            model_base = multi_gpu_model(model_base, gpus=num_gpus)

    model_params = package_lib.model_init(model_base,flavor,state_dir,
                                          optparams,**kwargs)
    model_params.setdefault('start_epoch',start_epoch)
    model_params.setdefault('start_monitor',start_monitor)
    model_params.setdefault('transpose',model_transpose)
    model_params.setdefault('rtranspose',model_rtranspose)
    model_params.setdefault('input_shape',input_shape)
    model_params.setdefault('pid',model_pid)
    model_params.setdefault('val_monitor','val_loss')
    model = Model(**model_params)
    
    model_png = pathjoin(model_dir,'model_pid%d.png'%model_pid)
    if pathexists(model_png):
        os.remove(model_png) # delete the old png to avoid irritating warnings
    try:
        from keras.utils import plot_model        
        plot_model(model.base, to_file=model_png, show_layer_names=True,
                   show_shapes=True)
        print('Saved model diagram to "%s"'%model_png)
    except Exception as e:
        from keras.utils import print_summary
        warn('Unable to generate model plot "%s" due to exception: %s'%(model_png,
                                                                        str(e)))
        print('Saving text-based model instead')
        model_txt = pathjoin(model_dir,'model_layers_pid%d.txt'%model_pid)
        with open(model_txt,'w') as fid:
            print2fid = lambda *args: print(''.join(args),file=fid)
            print_summary(model.base,print_fn=print2fid)

    model_summary(model.base)
            
    print('Compiling',flavor,'model')
    model.compile()
    
    model.initialized = True        
    print('Model',flavor,'initialized')
    
    return model

if __name__ == '__main__':
    print('hello')
   
