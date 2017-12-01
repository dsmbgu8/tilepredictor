from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import time
gettime = time.time

from pylib import *

pyext=expanduser('~/Research/src/python/external')
#sys.path.insert(0,pathjoin(pyext,'keras2/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras204/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras207/build/lib'))
#sys.path.insert(0,pathjoin(pyext,'keras208/build/lib'))
sys.path.insert(0,pathjoin(pyext,'keras-multiprocess-image-data-generator'))
sys.path.insert(0,pathjoin(pyext,'CLR'))

from pylib.dnn import *

try:
    from clr_callback import CyclicLR
    use_clr = True
except:
    print('Could not import CyclicLR callback!')
    use_clr = False
    raw_input()
    
from tilepredictor_util import *

tilepredictor_path = abspath(pathsplit(__file__)[0])
valid_packages = ['keras']
model_path = pathjoin(tilepredictor_path,'models')
valid_flavors = []
for pkg in valid_packages:
    for path in glob(pathjoin(model_path,'*'+pkg+'*.py')):
        model_dir,flavor_file = pathsplit(path)
        valid_flavors.append(flavor_file.split('_')[0])
                          
valid_flavors = list(set(valid_flavors))

default_package   = valid_packages[0] # 'keras'
default_flavor    = valid_flavors[0] # 'cnn3'
default_state_dir = './state/'

# softmax probs are:
# [0.0,0.5] class 0
# [0.5,1.0] class 1
# by default, we map these to [0.0,0.5] and store the class label separately
# scale_probs maps the [0.0,0.5] probs to [0.0,1.0] per class
scale_probs = True
randomize = False

# network architecture / training parameters
nb_hidden = 1024 # nodes in last FC layer before classification
nb_classes = 2 # we currently only consider binary classification problems 
output_shape = [nb_hidden,nb_classes]

# optimizer parameters
nb_epochs = 5000
batch_size = 32
nb_workers = 1
random_state = 42
tol = 1e-8
optparams = dict(
    weight_decay = 1e-6,
    reduce_lr = 0.1,
    step_lr = 100,
    beta_1 = 0.9,
    beta_2 = 0.999,
    lr_base = 0.0001,
    lr_max = 0.001,
    obj_lambda2 = 0.0025,
    max_norm = np.inf, # 5.0
    tol = tol,
    stop_delta = 0.001
)

train_rot_range=180.0
train_shear_range=10.0 # shear degrees
train_shift_range=0.1 # percentage of rows/cols to shift
train_zoom_range=0.1 # range = (1-zoom,1+zoom)
train_imaugment_params = dict(
    zoom_range = (1.0-train_zoom_range, 1.0+train_zoom_range),
    rotation_range = (-train_rot_range, train_rot_range),
    shear_range = (-train_shear_range, -train_shear_range),
    translation_range = (-train_shift_range, train_shift_range),
)

train_datagen_params = dict(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            zca_epsilon=tol,
                            rotation_range=train_rot_range,
                            width_shift_range=train_shift_range,
                            height_shift_range=train_shift_range,
                            shear_range=np.deg2rad(train_shear_range),
                            zoom_range=train_zoom_range,
                            fill_mode='wrap',
                            horizontal_flip=True,
                            vertical_flip=True,
                            rescale=None,
                            preprocessing_function=None)

@threadsafe_generator
def datagen_arrays(X,y,batch_size,datagen_params=train_datagen_params,
                   shuffle=True,fill_partial=True,
                   random_state=random_state,verbose=0,
                   preprocessing_function=None):
    from keras.preprocessing.image import ImageDataGenerator
    
    # only call datagen.fit() if these keys are present
    fit_kw = ['zca_whitening','featurewise_center',
              'featurewise_std_normalization']
        
    flowkw = dict(batch_size=batch_size,shuffle=shuffle,seed=random_state,
                  save_to_dir=None,save_prefix='',save_format='png')
    datagen_params.setdefault('preprocessing_function',preprocessing_function)
    datagen = ImageDataGenerator(**datagen_params)
    # only fit datagen if one of the fit keys is true
    if any([datagen_params.get(key,False) for key in fit_kw]):
        print('Fitting ImageDataGenerator for %d samples (this could take awhile)'%len(X))
        fittime = gettime()
        datagen.fit(X,seed=random_state)
        print('Fit complete, processing time: %0.3f seconds'%gettime()-fittime)
    transform = datagen.random_transform
    datagen_iter = datagen.flow(X,y,**flowkw)
    for bi, (X_batch, y_batch) in enumerate(datagen_iter):
        bi_collection = is_collection(X_batch)
        if bi_collection:
            X_batch = X_batch.concatenate()
        
        if fill_partial and X_batch.shape[0] < batch_size:
            # fill a partial batch with balanced+transformed inputs
            X_fill,y_fill = fill_batch(X_batch,y_batch,batch_size,
                                       balance=True)
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

@threadsafe_generator
def datagen_directory(path,target_size,batch_size,
                      datagen_params=train_datagen_params,
                      classes=None,class_mode='categorical',                      
                      shuffle=True,fill_partial=True,
                      random_state=random_state,
                      preprocessing_function=None,
                      verbose=0):
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
        
        if fill_partial and X_batch.shape[0] < batch_size:
            # fill a partial batch with balanced+transformed inputs
            X_fill,y_fill = fill_batch(X_batch,y_batch,batch_size,
                                       balance=True)
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

        self.callbacks     = []
        self.start_epoch   = kwargs['start_epoch']
        self.start_monitor = kwargs['start_monitor']
        self.transpose     = kwargs['transpose']

        self.val_monitor   = 'val_loss'
        self.val_type      = self.val_monitor.replace('val_','')
        self.val_best      = None
        self.val_cb        = None
        
        model_suf = '_'.join([self.flavor,self.package])
        self.model_dir = pathjoin(self.state_dir,model_suf)
        if not pathexists(self.model_dir):
            makedirs(self.model_dir,verbose=True)

    def preprocess(self,img,transpose=True,verbose=0):
        n_bands=img.shape[-1]
        dtype = img.dtype
        if dtype != np.uint8:
            raise Exception('No preprocessing function defined for dtype "%s"!'%str(dtype))
        shape = img.shape        
        if img.ndim not in (3,4) or n_bands != 3:
            raise Exception('No preprocessing function defined for image shape "%s"!'%str(shape))
        imgpre = preprocess_img_u8(img)
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
        kwargs.setdefault('overwrite',True)
        self.base.save(modelf,**kwargs)
    
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

    def update_validation_callback(self,test_data,test_labs,test_ids,**kwargs):
        if not self.val_cb:
            warn('validation callback not initialized')
            return

        self.val_cb.update_data(test_data,test_labs,test_ids,**kwargs)            

    def init_callbacks(self,nb_epochs=nb_epochs,**kwargs):
        from keras.callbacks import TerminateOnNaN, EarlyStopping, \
            ReduceLROnPlateau, CSVLogger, TensorBoard
        from validation_checkpoint import ValidationCheckpoint
        
        val_monitor = kwargs.pop('monitor','val_loss')
        step_lr = kwargs.pop('step_lr',optparams['step_lr'])
        step_lr = step_lr or min(100,max(1,int(nb_epochs*0.01)))
        stop_early = kwargs.pop('stop_early',max(10*step_lr,int(nb_epochs*0.2)))
        
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
        
        # configure callbacks
        train_logf = pathjoin(model_dir,'training_log.csv')
        val_mode = 'auto'

        self.val_monitor = val_monitor
        self.save_preds = save_preds
        self.save_model = save_model
        self.val_period = save_epoch
        self.verbose_callbacks = verbose
            
        self.val_cb = ValidationCheckpoint(val_monitor=val_monitor,
                                           save_best_preds=save_preds,
                                           save_best_model=save_model,
                                           model_dir=model_dir,
                                           mode=val_mode,
                                           initial_monitor=initial_monitor,
                                           period=save_epoch, verbose=verbose)
        #self.val_cb = ModelCheckpoint(model_iterf,monitor=val_monitor,mode=val_mode, period=save_epoch,
        #                        save_best_only=True, save_weights_only=False,                                
        #                        verbose=False)
        if use_clr:
            self.lr_cb = CyclicLR(base_lr=optparams['lr_base'],
                                  max_lr=optparams['lr_max'],
                                  step_size=step_lr)
        else:
            self.lr_cb = ReduceLROnPlateau(monitor=val_monitor,
                                           mode=val_mode,
                                           patience=step_lr,
                                           min_lr=optparams['lr_base'],
                                           factor=optparams['reduce_lr'],
                                           epsilon=optparams['tol'],
                                           verbose=verbose)
        
        self.es_cb = EarlyStopping(monitor=val_monitor, patience=stop_early,
                                   mode=val_mode, min_delta=optparams['stop_delta'],
                                   verbose=verbose)
        self.tn_cb = TerminateOnNaN()
        if pathexists(train_logf) and pathsize(train_logf) != 0:
            ctimestr = epoch2str(pathctime(train_logf))
            logf_base,logf_ext = splitext(train_logf)
            old_logf = logf_base+'_'+ctimestr+logf_ext
            print('Backing up existing log file "%s" to "%s"'%(train_logf,old_logf))
            os.rename(train_logf,old_logf)
        
        self.cv_cb = CSVLogger(filename=train_logf,append=True)
        self.callbacks = [self.val_cb,self.lr_cb,self.es_cb,self.tn_cb,self.cv_cb]

        use_tb = False
        if use_tb:
            tb_batch_size=32
            tb_freq = 10
            tb_log_dir = pathjoin(model_dir,'tb_logs')
            self.tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=tb_freq,
                                batch_size=tb_batch_size,
                                write_graph=True, write_grads=True,
                                write_images=True, embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)
            self.callbacks.append(self.tb_cb)
                
    def write_mispreds(self, outf, mispred_ids):
        n_mispred=len(mispred_ids)
        if n_mispred==0:
            return
        with open(outf,'w') as fid:
            print('# %d mispreds'%n_mispred,file=fid)
            for i,m_id in enumerate(mispred_ids):
                print('%s'%(str(m_id)),file=fid)

    def imaugment_batches(self,X_train,y_train,n_batches,
                         random_state=random_state):
        from sklearn.model_selection import StratifiedShuffleSplit        
        
        batch_transpose = [0]+[i+1 for i in self.transpose]
            
        # compute batch size wrt number of augmented samples
        batch_size = int(len(X_train)/n_batches)
        #batch_step = max(3,int(n_batches/10))
        #test_batch = int(n_batches // batch_step)
        #test_batch_idx = np.linspace(test_batch,n_batches-test_batch,batch_step)                
        #test_batch_idx = np.unique(np.round(test_batch_idx))
        split_state = random_state
        try:
            while(1):
                sss = StratifiedShuffleSplit(n_splits=n_batches,
                                             train_size=batch_size,
                                             random_state=split_state)
                split_state = split_state+1
                X_batch,y_batch = [],[]
                X_train_batch,y_train_batch = [],[]
                for bi,(batch_idx,_) in enumerate(sss.split(y_train,y_train)):                
                    X_batch,y_batch = collect_batch(X_train,y_train,
                                                    batch_idx=batch_idx,
                                                    imgs_out=X_batch,
                                                    labs_out=y_batch)

                    X_train_batch,y_train_batch = imaugment_perturb(X_batch,y_batch,
                                                                    imgs_out=X_train_batch,
                                                                    labs_out=y_train_batch)

                    yield X_train_batch.transpose(batch_transpose), y_train_batch
                    
        except KeyboardInterrupt:
            if self.initialized:
                print('User interrupt, saving model')
                #self.save_weights(out_weightf)
                
    def train(self,train_gen,n_epochs,n_batches,initial_epoch=None,
              validation_data=None,verbose=1,**kwargs):

        if len(self.callbacks)==0:
            warn('Training model with no callbacks (did you call init_callbacks?)')

        initial_epoch = initial_epoch or self.start_epoch
            
        self.base.fit_generator(train_gen,n_batches,
                                epochs=n_epochs,
                                initial_epoch=initial_epoch,
                                validation_data=validation_data,
                                callbacks=self.callbacks,
                                workers=nb_workers,                                
                                verbose=verbose)

        self.initialized = True

def compile_model(input_shape,output_shape,**params):
    from keras.backend import image_data_format,set_image_data_format
    import importlib
    
    nb_hidden,nb_classes = output_shape

    package   = params.pop('model_package',default_package)
    flavor    = params.pop('model_flavor',default_flavor)
    state_dir = params.pop('model_state_dir',None)
    weightf   = params.pop('model_weightf',None)
    flavorp   = params.pop('flavor_params',{})

    use_backend_format = kwargs.pop('use_backend_format',True)
    
    # new paths: e.g., state_dir/cnn3_keras
    state_suf = '_'.join([flavor,package])
    if pathexists(pathjoin(state_dir,package,flavor)):
        # old paths: e.g., state_dir/keras/cnn3
        state_suf = pathjoin(package,flavor)
        
    if weightf and not pathexists(weightf):
        if state_dir and pathexists(pathjoin(state_dir,state_suf,weightf)):
            print('Found weight file "%s" in state_dir "%s"'%(weightf,state_dir))
            weightf = pathjoin(state_dir,state_suf,weightf)
        else:
            print('Weight file "%s" not found'%weightf)
            weightf = None
    
    if not state_dir:
        if weightf:
            model_dir,weight_file = pathsplit(weightf)
            state_dir = model_dir.replace(state_suf,'')
            state_dir = state_dir.replace('//','/')
            print('Using model state_dir="%s"'%state_dir)
        else:
            state_dir = default_state_dir
    model_dir = pathjoin(state_dir,state_suf)
    num_gpus  = params.pop('num_gpus',0)

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

    model_transpose = [0,1,2]
    set_image_data_format('channels_last')
    if use_backend_format:
        if model_backend=='tensorflow':
            set_image_data_format('channels_last')
            if input_shape[0]==3:
               input_shape = input_shape[1:]+[3]
               model_transpose = [2,0,1]
        elif model_backend=='theano':    
            set_image_data_format('channels_first')
            if input_shape[-1]==3:
               input_shape = [3]+input_shape[:-1]
               model_transpose = [2,0,1]

    start_epoch, start_monitor = 0, None
    if weightf:
        start_epoch, start_monitor = parse_model_meta(weightf)
        print('Restoring existing %s_%s model'%(flavor,package),
              'from file: "%s"'%weightf,'with',
              'start_epoch=%d,'%start_epoch,
              'start_monitor=%.6f'%start_monitor)
        model_base = package_lib.load_model(weightf)        
    else:
        print('Initializing new %s_%s model'%(flavor,package),
              'with','input_shape=%s,'%str(input_shape),
              'model_transpose=%s,'%str(model_transpose),
              'image_data_format=%s'%image_data_format())
        model_params = flavor_lib.model_init(input_shape,**flavorp)
        lr_mult = model_params.pop('lr_mult',1.0)        
        model_base = model_params['model']

        if lr_mult!=1.0:
            lr_upkeys = []
            for key,val in optparams.iteritems():
                if key.startswith('lr_'):
                    optparams[key] = optparams[key]*lr_mult
                    lr_upkeys.append(key)

            print('Updated optparams "%s" with lr_mult=%6.3f'%(str(lr_upkeys),
                                                               lr_mult))
        
        model_base = package_lib.update_base_outputs(model_base,output_shape,
                                                     optparam=optparams)

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
                                          optparams,**params)
    model_params.setdefault('start_epoch',start_epoch)
    model_params.setdefault('start_monitor',start_monitor)
    model_params.setdefault('transpose',model_transpose)
    model_params.setdefault('input_shape',input_shape)
    model = Model(**model_params)
    
    model_png = pathjoin(model_dir,'model.png')
    if pathexists(model_png):
        os.remove(model_png) # delete the old png to avoid irritating warnings
    try:
        from keras.utils.vis_utils import plot_model        
        plot_model(model.base, to_file=model_png, show_layer_names=True,
                   show_shapes=True)
        print('Saved model diagram to "%s"'%model_png)
    except Exception as e:
        warn('Unable to generate model diagram "%s" due to exception: %s'%(model_png,
                                                                           str(e)))
    print('Compiling',flavor,'model')

    model.compile()
    
    model.initialized = True        
    print('Model',flavor,'initialized')
    
    return model

