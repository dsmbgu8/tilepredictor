from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import time
gettime = time.time

from util.aliases import *

#sys.path.insert(0,expanduser('~/Downloads/keras2/build/lib/'))
#sys.path.insert(0,expanduser('~/Downloads/keras204/build/lib/'))
sys.path.insert(0,expanduser('~/Research/src/python/external/keras204/build/lib/'))
sys.path.insert(1,expanduser('~/Research/src/python/external/keras-multiprocess-image-data-generator'))

from util.aliases.dnn import *

from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback

from tilepredictor_util import *

tilepredictor_path = abspath(pathsplit(__file__)[0])
valid_packages = ['keras']
model_path = pathjoin(tilepredictor_path,'models')
valid_flavors = []
for pkg in valid_packages:
    model_list = glob(pathjoin(model_path,'*'+pkg+'*.py'))
    valid_flavors.extend([pathsplit(modelpath)[1].split('_')[0]
                          for modelpath in model_list])
valid_flavors = list(set(valid_flavors))

default_package   = valid_packages[0]
default_flavor    = valid_flavors[0]
default_state_dir = './state'

# network architecture / training parameters
nb_classes = 2 # we currently only consider binary classification problems 
nb_hidden = 1024
nb_epochs = 10000
random_state = 42
max_norm = np.inf # 5.0
obj_lambda2 = 0.0025

# optimizer parameters
tol = 1e-8
optparams = dict(
    init_lr = 0.002,
    min_lr = tol,
    weight_decay = 0.004,
    lr_reduce= 0.2,
    beta_1 = 0.9,
    beta_2 = 0.999,
    tol = tol
)

batch_transform_params = dict(
    zoom_range = (1.0, 1.1),
    rotation_range = (0., 180.),
    shear_range = (-10., -10.),
    translation_range = (-10, 10),
)

msort = ['fscore','precision','recall']
def compute_metrics(test_lab,pred_lab,pos_label=1,average='binary'):
    prfs = precision_recall_fscore_support(test_lab,pred_lab,average=average,
                                           pos_label=pos_label)
    return dict(zip(['precision','recall','fscore'],prfs[:-1]))

def prediction_summary(test_lab,pred_lab,npos,nneg,best_fs,best_epoch):
    mout = compute_metrics(test_lab,pred_lab)
    neg_preds = pred_lab==0
    pos_preds = pred_lab==1
    err_preds = pred_lab!=test_lab

    nneg_pred = np.count_nonzero(neg_preds)
    npos_pred = np.count_nonzero(pos_preds)
    mtup = (npos,npos_pred,nneg,nneg_pred)

    npos_mispred = np.count_nonzero(pos_preds & err_preds)
    nneg_mispred = np.count_nonzero(neg_preds & err_preds)
 
    mstr = ', '.join(['%s=%9.6f'%(m,mout[m]*100) for m in msort])
    mstr += '\n     pos=%d, pos_pred=%d, neg=%d, neg_pred=%d'%mtup
    mstr += '\n     mispred pos=%d, neg=%d'%(npos_mispred,nneg_mispred)
    mstr += '\n     best fscore=%9.6f, epoch=%d'%(best_fs*100,best_epoch)
    return mout,mstr

def write_predictions(predf, test_ids, test_lab, pred_lab, pred_out, pred_mets,
                      fprfnr=True, buffered=False):
    if len(test_ids)==[]:
        warn('cannot write predictions without test_ids')
        return
    
    mout  = pred_mets or compute_metrics(test_lab,pred_lab)
    mstr  = ', '.join(['%s=%9.6f'%(m,mout[m]*100) for m in msort])

    if fprfnr:
        from scipy import interp
        from sklearn.metrics import roc_curve
        pred_prob = np.where(pred_lab==1,pred_out,1.0-pred_out)
        fprs, tprs, thresholds = roc_curve(test_lab, pred_prob)
        fnrs = 1.0-tprs
        optidx = np.argmin(fprs+fnrs)
        fpr,fnr = fprs[optidx]*100,fnrs[optidx]*100
        
        fprs_interp = np.linspace(0.0, 1.0, 101)
        fnrs_interp = interp(fprs_interp, fprs, fnrs)
        fnr1fpr = fnrs_interp[fprs_interp==0.01][0]*100
        mstr += '\n# fpr=%9.6f, fnr=%9.6f, fnr@1%%fpr=%9.6f'%(fpr,fnr,fnr1fpr) 
    
    n_lab = len(test_ids)
    m_err = test_lab!=pred_lab
    pos_lab,neg_lab = (test_lab==1),(test_lab!=1)
    n_tp = np.count_nonzero(~m_err & pos_lab)
    n_tn = np.count_nonzero(~m_err & neg_lab)
    n_fp = np.count_nonzero(m_err & neg_lab)
    n_fn = np.count_nonzero(m_err & pos_lab)    
    n_acc,n_err = n_tp+n_tn, n_fp+n_fn
    n_pos,n_neg = n_tp+n_fn, n_tn+n_fp
    outstr = ['# %d samples: # %d correct, %d errors'%(n_lab,n_acc,n_err),
              '# [tp=%d+fn=%d]=%d positive'%(n_tp,n_fn,n_pos) + \
              ' [tn=%d+fp=%d]=%d negative'%(n_tn,n_fp,n_neg),
              '# %s'%mstr, '#', '# id lab pred prob']
    with open(predf,'w') as fid:
        if not buffered:
            print('\n'.join(outstr),file=fid)        
        for i,m_id in enumerate(test_ids):
            labi,predi,probi = test_lab[i],pred_lab[i],pred_out[i]
            outstri = '%s %d %d %7.4f'%(str(m_id),labi,predi,probi*100)
            if buffered:
                outstr.append(outstri)
            else:
                print(outstri,file=fid)
        if buffered:
            print('\n'.join(outstr),file=fid)

class MetricsCheckpoint(Callback):
    global msort
    def __init__(self,model_dir,metrics=msort,val_monitor='val_loss',
                 mode='auto',test_ids=[],save_best_preds=False,
                 period=1,verbose=0):
        super(MetricsCheckpoint, self).__init__()
        self.metrics     = dict([(m,[]) for m in metrics])
        self.test_lab    = None
        self.statestr    = None
        
        self.val_monitor = val_monitor
        self.val_mode    = mode
        self.val_func    =  np.max if self.val_mode=='max' else np.min
        self.val_best    = -np.inf if self.val_mode=='max' else np.inf

        self.test_ids    = test_ids
        self.save_preds  = save_best_preds
        self.period      = period
        self.verbose     = verbose
        self.model_dir   = model_dir
        
    def on_train_begin(self, logs=None):
        logs = logs or {}

    def on_train_end(self, logs=None):
        logs = logs or {}

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        
    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_start_time  = gettime()
        if self.statestr:
            print(self.statestr)
            self.statestr = None
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_end_time  = gettime()
        etime = self.epoch_end_time-self.epoch_start_time
        val_monitor = self.val_monitor
        statestr = '\nEpoch %05d: processing time: %0.3f seconds'%(epoch+1,etime)
        if self.validation_data:
            if self.test_lab is None:
                self.test_lab = np.int8(np.argmax(self.validation_data[1],-1))

            pred_out = np.asarray(self.model.predict(self.validation_data[0]))
            pred_lab = np.int8(np.argmax(pred_out,axis=-1))
            pred_out = np.amax(pred_out,axis=-1)
            pred_mets = compute_metrics(self.test_lab,pred_lab)
            mstr = ['loss=%9.6f'%logs['val_loss']]
            for m in self.metrics:
                mval = pred_mets[m]
                self.metrics[m].append(mval)
                logs['val_'+m] = mval
                mstr.append('%s=%9.6f'%(m,pred_mets[m]*100))
            mstr = ', '.join(mstr)
            statestr += '\nValidation '+mstr #+', support='+str(supp)

            if (epoch % self.period)==0:
                val_epoch = np.float32(logs[val_monitor])
                val_cmp = self.val_func([self.val_best,val_epoch])            
                if val_cmp==val_epoch and val_cmp != self.val_best:
                    val_best = val_epoch
                    statestr += '\nNew best %s=%9.6f'%(val_monitor,val_best)
                    self.val_best = val_best
                    if self.save_preds:
                        monitor_str = val_monitor.replace('val_','')
                        predsf = 'preds_iter{epoch:d}_%s{%s:.6f}.txt'%(monitor_str,val_monitor)
                        predd = {'epoch':epoch,val_monitor:val_best}
                        predsf = pathjoin(self.model_dir,predsf.format(**predd))
                        statestr += '\nSaved best predictions to %s'%predsf
                        write_predictions(predsf, self.test_ids, self.test_lab,
                                          pred_lab, pred_out, pred_mets,
                                          fprfnr=True, buffered=False)
        statestr += '\n'
        self.statestr = statestr

#@timeit
def collect_batch(imgs,labs,batch_idx=[],imgs_out=[],labs_out=[]):
    # collect img_out,lab_out from collection imgs
    imgshape = imgs[0].shape
    nbatch = len(batch_idx)
    if nbatch==0:
        nbatch = len(labs)
        batch_idx = range(nbatch)
    if len(imgs_out)!=nbatch:
        imgs_out = np.zeros([nbatch]+list(imgshape),dtype=imgs[0].dtype)
        labs_out = np.zeros([nbatch,labs.shape[1]],dtype=labs[0].dtype)
    for i,idx in enumerate(batch_idx):
        imgs_out[i] = imgs[idx]
        labs_out[i] = labs[idx]
    return imgs_out,labs_out

#@timeit
def perturb_batch(*args,**kwargs):
    from imaugment import perturb_batch as _pb
    kwargs['train_params'] = batch_transform_params
    return _pb(*args,**kwargs)

class Model(object):
    """
    Model: wrapper class for package model predictor functions
    """
    def __init__(self, **kwargs):
        self.initialized  = False        
        self.flavor       = kwargs['flavor']
        self.package      = kwargs['package']
        self.backend      = kwargs['backend']
        self.params       = kwargs['params']
        self.base         = kwargs['base']
        self.batch        = kwargs['batch']
        self.predict      = kwargs['predict']
        self.transform    = kwargs['transform']
        self.saveweights  = kwargs['save']
        self.loadweights  = kwargs['load']
        self.compile      = kwargs['compile']
        self.preprocess   = preprocess_img_u8

        if self.backend=='tensorflow':
            self.image_data_format = 'channels_last' 
            self.transpose = (0,1,2) # channels last=(0,1,2)=default
        elif self.backend=='theano':
            self.image_data_format = 'channels_first' 
            self.transpose = (2,0,1) # channels first=(2,0,1)

        self.model_dir = pathjoin(kwargs['state_dir'],self.package,self.flavor)
        if not pathexists(self.model_dir):
            makedirs(self.model_dir,verbose=True)

        model_png = pathjoin(self.model_dir,'model.png')
        if not pathexists(model_png):
            from keras.utils.vis_utils import plot_model
            print('Saving model diagram to %s'%model_png)
            plot_model(self.base, to_file=model_png, show_layer_names=True,
                       show_shapes=True)            

    def save(self,outfile):
        pass
    
    def load(self,infile):
        pass
            
    def save_weights(self, *args, **kwargs):
        return self.saveweights(*args,**kwargs)
    
    def load_weights(self, *args, **kwargs):
        self.base = self.loadweights(*args,**kwargs)
        self.initialized = True

    def write_mispreds(self, outf, mispred_ids):
        n_mispred=len(mispred_ids)
        if n_mispred==0:
            return
        with open(outf,'w') as fid:
            print('# %d mispreds'%n_mispred,file=fid)
            for i,m_id in enumerate(mispred_ids):
                print('%s'%(str(m_id)),file=fid)

    def generate_batches(self,X_train,y_train,n_batches,
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

                    X_train_batch,y_train_batch = perturb_batch(X_batch,y_batch,
                                                                imgs_out=X_train_batch,
                                                                labs_out=y_train_batch)

                    yield X_train_batch.transpose(batch_transpose), y_train_batch
        except KeyboardInterrupt:
            if self.initialized:
                print('User interrupt')
                #print('saving model')
                #out_weightf = train_weight_iterf%epoch
                #self.save_weights(out_weightf)
                
    def fit_generator(self,*args,**kwargs):
         return self.base.fit_generator(*args,**kwargs)
            
    def train(self,X_train,y_train,X_test=[],y_test=[],
              nb_epochs=nb_epochs,**kwargs):
        
        from keras.callbacks import TerminateOnNaN, EarlyStopping, \
            ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
        
        batch_size = kwargs.pop('batch_size',128)
        augment = kwargs.pop('augment',1.0)
        test_percent = kwargs.pop('test_percent',0.2)
        # strides to test/save model during training
        test_epoch = kwargs.pop('test_epoch',1)
        save_epoch = kwargs.pop('save_epoch',1)
        random_state = kwargs.pop('random_state',42)
        collect_test = kwargs.pop('collect_test',True)

        val_monitor = kwargs.pop('val_monitor','val_loss')
        verbose = kwargs.pop('verbose',1)

        # exit early if the last [stop_early] test scores are all worse than the best
        stop_early = kwargs.pop('stop_early',max(100,int(nb_epochs*0.2)))
        lr_step = kwargs.pop('lr_step',max(10,int(nb_epochs*0.01)))

        save_preds = kwargs.pop('save_preds',True)
        train_ids = kwargs.pop('train_ids',[])
        test_ids = kwargs.pop('test_ids',[])
            
        if y_train.ndim==1 or y_train.shape[1]==1:
            train_lab = y_train.copy()
            y_train = to_categorical(y_train, nb_classes)
        else:
            train_lab = np.argmax(y_train,axis=-1)

        print("Training samples: {}, input shape: {}".format(len(X_train),X_train[0].shape))
        print('Training classes: {}'.format((np.count_nonzero(train_lab==0),np.count_nonzero(train_lab==1))))            
        if len(y_test)>0:
            if y_test.ndim==1 or y_test.shape[1]==1:
                test_lab = y_test.copy()
                y_test = to_categorical(y_test, nb_classes)
            else:
                test_lab = np.argmax(y_test,axis=-1)
            
            n_neg,n_pos = np.count_nonzero(test_lab==0),np.count_nonzero(test_lab==1)
            print("Testing samples: {}, input shape: {}".format(len(X_test),X_test[0].shape))        
            print('Testing classes: {} neg, {} pos'.format(n_neg,n_pos))

            if collect_test:
                print('Loading %d test samples into memory'%(len(y_test)))
                X_test,y_test = collect_batch(X_test,y_test)
                for bi in range(X_test.shape[-1]):
                    bmin,bmax=extrema(X_test[...,bi].ravel())
                    print('band[%d] min=%.3f, max=%.3f'%(bi,bmin,bmax))

                batch_transpose = [0]+[dimv+1 for dimv in self.transpose]
                X_test = X_test.transpose(batch_transpose)
            validation_data = (X_test,y_test)
        else:
            print("Testing samples: 0")
            print('Testing classes: 0 neg, 0 pos')
            
            n_neg,n_pos = 0,0
            val_monitor = val_monitor.replace('val_','')
            validation_data = None

            
        model_dir = self.model_dir
        
        # configure callbacks
        monitor_str = val_monitor.replace('val_','')
        model_iterf  = 'model_iter{epoch:d}_%s{%s:.6f}.h5'%(monitor_str,val_monitor)
        model_iterf = pathjoin(model_dir,model_iterf)
        train_csv_logf = pathjoin(model_dir,'training.log')
        val_mode='max' if monitor_str in ('fscore','acc') else 'min'

        em_cb = MetricsCheckpoint(model_dir,save_best_preds=save_preds,
                                  test_ids=test_ids, metrics=msort, 
                                  val_monitor=val_monitor, mode=val_mode,
                                  period=save_epoch, verbose=verbose)
        cp_cb = ModelCheckpoint(model_iterf,monitor=val_monitor,mode=val_mode, period=save_epoch,
                                save_best_only=True, save_weights_only=False,                                
                                save_first=False,verbose=False)
        lr_cb = ReduceLROnPlateau(monitor=val_monitor, mode=val_mode,
                                  patience=lr_step, min_lr=optparams['min_lr'],
                                  factor=optparams['lr_reduce'],
                                  epsilon=optparams['tol'],
                                  verbose=verbose)
        es_cb = EarlyStopping(monitor=val_monitor, patience=stop_early,
                              mode=val_mode, min_delta=0.01, verbose=verbose)
        tn_cb = TerminateOnNaN()
        cv_cb = CSVLogger(filename=train_csv_logf,append=False)
        cb_list = [em_cb,cp_cb,lr_cb,es_cb,tn_cb,cv_cb]

        use_tb=False
        if use_tb:
            tb_freq = 10
            tb_log_dir = pathjoin(model_dir,'tb_logs')
            tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=tb_freq,
                                batch_size=batch_size,
                                write_graph=True, write_grads=True,
                                write_images=True, embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)
            cb_list.append(tb_cb)

        n_batches = int(len(X_train)//batch_size)
        msg = ['Training network for %d epochs'%nb_epochs,
               'batch size=%d'%batch_size,
               'batches/epoch=%d'%n_batches]
        print(', '.join(msg))

        nb_workers = 1
        batch_gen = self.generate_batches(X_train,y_train,n_batches)
        self.fit_generator(batch_gen,n_batches,validation_data=validation_data,
                           epochs=nb_epochs,workers=nb_workers,
                           callbacks=cb_list,verbose=verbose)

        self.initialized = True

def compile_model(input_shape,nb_classes,**params):
    package   = params.pop('model_package',default_package)
    flavor    = params.pop('model_flavor',default_flavor)
    state_dir = params.pop('model_state_dir',default_state_dir)
    weightf   = params.pop('model_weightf',None)
    
    if package not in valid_packages:
        warn('Invalid model_package: %s (valid options=%s)'%(package,str(valid_packages)))
        sys.exit(1)

    if flavor not in valid_flavors:
        warn('Invalid model_flavor: %s (valid options=%s)'%(flavor,str(valid_flavors)))
        sys.exit(1)
        
    if package=='keras':
        import keras_model
        load_model = keras_model.load_model
        model_init = keras_model.model_init
        update_base_outputs = keras_model.update_base_outputs

        #  NOTE (BDB, 09/12/17): there must be a better way to do this... 
        if flavor == 'cnn3':        
            from models.cnn3_keras import model_init as init_flavor
        elif flavor == 'cnn5':
            from models.cnn5_keras import model_init as init_flavor
        elif flavor == 'ffstn':
            from models.ffstn_keras import model_init as init_flavor
        elif flavor in ('inception','inceptionv3'):
            from models.inception_keras import model_init as init_flavor
        elif flavor in ('xception','xceptionv1'):
            from models.xception_keras import model_init as init_flavor        
    elif package=='lasagne':
        import lasagne_model
        load_model = lasagne_model.load_model
        model_init = lasagne_model.lasagne_model
        if flavor == 'cnn3':
            from models.cnn3_lasagne import model_init as init_flavor        
        elif flavor == 'ffstn':
            from models.ffstn_lasagne import model_init as init_flavor

    if weightf:
        print('Restoring existing model from file:',weightf)
        model_base = load_model(weightf)        
    else:
        model_base = init_flavor(input_shape,**params)
        model_base = update_base_outputs(model_base,nb_classes,nb_hidden,
                                         obj_lambda2,max_norm)
    
    model_params = model_init(model_base,flavor,state_dir,optparams,**params)
    model = Model(**model_params)
    
    if weightf:
        model.initialized = True
    else:
        print('Compiling model')
        model.compile()
        
    print('Model+functions compiled for package',package,'flavor',flavor)
    
    return model

