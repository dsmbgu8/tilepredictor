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
from keras.callbacks import Callback as KerasCallback

valid_model_packages = ['keras','lasagne']
valid_model_flavors = ['cnn3','ffstn','inceptionv3','xception']

default_model_package = valid_model_packages[0]
default_model_flavor =  valid_model_flavors[0]

# learning parameters
nb_classes = 2 # we currently only consider binary classification problems 
nb_epochs = 10000

# exit early if the last [stop_early] test scores are all worse than the best
stop_early = 25

update_lr = 0.0002
weight_decay = 0.0
beta_1 = 0.9
beta_2 = 0.999
tol = 1e-5

# naugpos:naugneg = ratio of pos/neg augmented examples
naugpos = 1
naugneg = 1

# filename templates
weight_iterscoref = 'weights_iter%d_fscore%s.h5'
weight_iterf = 'weights_iter%d.h5'
layer0_iterf = 'layer0_iter%d.pdf'

plot_test_output=False # plot transformed examples
plot_mispreds=False # only plot mispreds
plot_augmented=False

nplotclass = 3 # number of examples/mispreds to plot per class

msort = ['precision','recall','fscore']
def compute_metrics(test_lab,pred_lab,pos_label):
    prfs = precision_recall_fscore_support(test_lab,pred_lab,
                                           pos_label=pos_label)
    return dict(zip(msort,prfs[:-1])) 

def prediction_summary(test_lab,pred_lab,ntp,ntn,best_fs,best_epoch):
    mout = compute_metrics(test_lab,pred_lab,pos_label=1)
    neg_preds = pred_lab==0
    pos_preds = pred_lab==1
    err_preds = pred_lab!=test_lab

    nn_pred = np.count_nonzero(neg_preds)
    np_pred = np.count_nonzero(pos_preds)
    mtup = (ntp,np_pred,ntn,nn_pred)

    np_mispred = np.count_nonzero(pos_preds & err_preds)
    nn_mispred = np.count_nonzero(neg_preds & err_preds)
    etup = (np_mispred,nn_mispred)

    mstr = ', '.join(['%s=%9.6f'%(m,mout[m]*100) for m in msort])
    mstr += '\n     tpos=%d, pos_pred=%d, tneg=%d, neg_pred=%d'%mtup
    mstr += '\n     mispred pos=%d, neg=%d'%etup
    mstr += '\n     best fscore=%9.6f, epoch=%d'%(best_fs*100,best_epoch)
    return mout,mstr

def write_preds(predf, test_ids, test_lab, pred_lab, pred_out):
    n_ids=len(test_ids)
    n_mispred = np.count_nonzero(test_lab!=pred_lab)
    with open(predf,'w') as fid:
        print('# %d/%d mispreds'%(n_mispred,n_ids),file=fid)
        print('# id lab pred prob',file=fid)
        for i,m_id in enumerate(test_ids):
            labi,predi = test_lab[i],pred_lab[i]
            probi = pred_out[i][int(predi)]
            outstri = '%s %d %d %7.4f'%(str(m_id),labi,predi,probi)
            print(outstri,file=fid)

        
@timeit
def collect_batch(imgs,labs,batch_idx=[],imgs_out=[],labs_out=[]):
    # collect img_out,lab_out from collection imgs
    imgshape = imgs[0].shape
    nbatch=len(batch_idx)
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

@timeit
def perturb_batch(*args,**kwargs):
    from imaugment import perturb_batch as _pb

    batch_transform_params = {
        'zoom_range': (1.0, 1.1),
        'rotation_range': (0., 360.),
        'shear_range': (-10., -10.),
        'translation_range': (-5, 5),
    }
    
    kwargs['train_params'] = batch_transform_params.copy()
    return _pb(*args,**kwargs)


class Model(object):
    """
    Model: wrapper class for package model predictor functions
    """
    def __init__(self, **kwargs):
        self.initialized = False        
        self.flavor      = kwargs['flavor']
        self.package     = kwargs['package']
        self.backend     = kwargs['backend']
        self.params      = kwargs['params']
        self.base        = kwargs['base']
        self.batch       = kwargs['batch']
        self.predict     = kwargs['predict']
        self.transform   = kwargs['transform']
        self.saveweights = kwargs['save']
        self.loadweights = kwargs['load']
        self.transpose   = None
        
    def save_weights(self, *args, **kwargs):
        return self.saveweights(*args,**kwargs)
    
    def load_weights(self, *args, **kwargs):
        ret = self.loadweights(*args,**kwargs)
        self.initialized = True
        return ret
        
    def train_test(self,X_train,y_train,X_test=[],y_test=[],**kwargs):
        from sklearn.model_selection import StratifiedShuffleSplit

        state_dir  = kwargs.pop('state_dir','./model_state')
        nb_epochs  = kwargs.pop('nb_epochs',1000)
        batch_size = kwargs.pop('batch_size',128)
        naugpos = kwargs.pop('naugpos',1)
        naugneg = kwargs.pop('naugneg',1)
        # strides to test/save model during training
        test_epoch = kwargs.pop('test_epoch',5)
        test_percent = kwargs.pop('test_percent',0.25)
        save_epoch = kwargs.pop('save_epoch',5)
        random_state = kwargs.pop('random_state',42)
        save_preds = kwargs.pop('save_preds',True)

        train_ids = kwargs.pop('train_ids',[])
        test_ids = kwargs.pop('test_ids',[])
        
        if y_train.ndim==1 or y_train.shape[1]==1:
            train_lab = y_train.copy()
            y_train = to_categorical(y_train, nb_classes)
        else:
            train_lab = np.argmax(y_train,axis=-1)
        
        if len(y_test)==0:
            from sklearn.model_selection import train_test_split as tts
            msg='No test data provided'
            msg+=', testing on %d%% of training data'%int(test_percent*100)
            warn(msg)
            n_train = X_train.shape[0]
            train_idx, test_idx = tts(np.arange(n_train),
                                      test_size=test_percent,
                                      stratify=train_lab,
                                      random_state=random_state)
            X_train, X_test = X_train[train_idx],X_train[test_idx]
            y_train, y_test = y_train[train_idx],y_train[test_idx]
            if len(train_ids)!=0:
                train_ids, test_ids = train_ids[train_idx],train_ids[test_idx]

        if y_test.ndim==1 or y_test.shape[1]==1:
            test_lab = y_test.copy()
            y_test = to_categorical(y_test, nb_classes)  
        else:
            test_lab = np.argmax(y_test,axis=-1)
            
        tnmask,tpmask = (test_lab==0),(test_lab==1)
        ntn,ntp = np.count_nonzero(tnmask),np.count_nonzero(tpmask)

        print("Training samples: {}, input shape: {}".format(len(X_train),X_train[0].shape))
        print('Training classes: {}'.format((np.count_nonzero(train_lab==0),np.count_nonzero(train_lab==1))))
        print("Testing samples: {}, input shape: {}".format(len(X_test),X_test[0].shape))        
        print('Testing classes: {}'.format((ntn,ntp)))
        
        model_dir = pathjoin(state_dir,self.package,self.flavor)
        if not pathexists(model_dir):
            makedirs(model_dir,verbose=True)
        
        train_weight_iterscoref = pathjoin(model_dir,weight_iterscoref)
        train_weight_iterf = pathjoin(model_dir,weight_iterf)
        train_layer0_iterf = pathjoin(model_dir,layer0_iterf)
        best_weightf = None

        if save_preds and len(test_ids)==0:
            warn('Cannot save preds without provided test_ids')
            save_preds=False

        batch_transpose = None
        if self.transpose:
            batch_transpose = [0]+[i+1 for i in self.transpose]

        collect_test = True            
        if collect_test:
            print('Loading %d test samples into memory'%(len(y_test)))
            X_test,y_test = collect_batch(X_test,y_test)
            if batch_transpose:
                X_test = X_test.transpose(batch_transpose)
            
        print("Training network for %d epochs, batch size=%d"%(nb_epochs,
                                                               batch_size))
        epoch = 0
        try:
            test_idx = range(len(y_test))
            test_hist = np.zeros(stop_early)
            test_histidx = 0
            test_histdelta = max(5,int(round(0.25*stop_early)))
            best_fs = 0 # keep track of best f-score (always save first nonzero eval)
            best_epoch = 0

            n_batches = int(len(X_train)/batch_size)
            # compute batch size wrt number of augmented samples
            batch_augment_size = batch_size//(naugpos+naugneg)
            sss = StratifiedShuffleSplit(n_splits=n_batches,
                                         train_size=batch_augment_size,
                                         random_state=random_state)

            #predict_one = lambda Xi: np.argmax(self.predict(Xi[np.newaxis]))
            #pred_lab = np.int8(map(predict_one,X_test))

            X_batch,y_batch = [],[]
            X_train_batch,y_train_batch = [],[]

            if n_batches <= 10:
                test_batch_idx = np.int32([round(n_batches/2.0)])
            else:                
                batch_step = max(5,int(n_batches/10))
                    
                test_batch = int(n_batches // batch_step)
                test_batch_idx = np.linspace(test_batch,n_batches-test_batch,
                                             batch_step)                
                test_batch_idx = np.unique(np.round(test_batch_idx))
                
            save_batch_ids = False
            batch_error = False
                           
            for epoch in range(nb_epochs+test_histdelta):
                # X_train is unused in sss, so we just pass a placeholder instead
                best_batch_fs = 0.0
                eloss,etime = 0.0,0.0
                for bi,(batch_idx,_) in enumerate(sss.split(y_train,y_train)):
                    print('Epoch %d:'%epoch,'batch',bi+1,'of',n_batches)
                    
                    starttime  = gettime()
                    X_batch,y_batch = collect_batch(X_train,y_train,
                                                    batch_idx=batch_idx,
                                                    imgs_out=X_batch,
                                                    labs_out=y_batch)

                    X_train_batch,y_train_batch = perturb_batch(X_batch,y_batch,
                                                                naugpos=naugpos,
                                                                naugneg=naugneg,
                                                                imgs_out=X_train_batch,
                                                                labs_out=y_train_batch)

                    if batch_transpose:
                        X_train_batch = X_train_batch.transpose(batch_transpose)

                    #bloss = self.epoch(X_train_batch, y_train_batch, i=epoch)                        
                    bloss = self.batch(X_train_batch, y_train_batch)
                    btime = gettime()-starttime
                    if save_batch_ids:
                        batch_ids = train_ids[batch_idx]
                        batch_idf = pathjoin(model_dir,'batch%d_ids.txt'%(bi+1))
                        batch_hdr = '# batch %d: %d ids'%(bi+1,len(batch_ids))
                        np.savetxt(batch_idf,batch_ids,header=batch_hdr)
                                            
                    print('batch loss:',bloss,'processing time: %0.3f seconds'%btime)
                    if bloss!=bloss:
                        print('encountered nan batch loss, bailing out')
                        batch_error=True                        
                        break
                    eloss += bloss
                    etime += btime
                    if bi in test_batch_idx:
                        print('Epoch {0}: test accuracy after batch {1} update'.format(epoch,bi+1))
                        pred_out = self.predict(X_test)
                        pred_lab = np.int8(np.argmax(pred_out,-1))
                        mout,mstr = prediction_summary(test_lab,pred_lab,ntp,ntn,best_fs,best_epoch)
                        print('Test {0}'.format(mstr))
                        best_batch_fs = max(best_batch_fs,mout['fscore'])
                        
                # end batch loop                
                if batch_error:
                    break
                
                if epoch % test_epoch == 0:
                    print('Epoch {0}: computing test accuracy'.format(epoch))
                    pred_out = self.predict(X_test)
                    pred_lab = np.int8(np.argmax(pred_out,-1))
                    mout,mstr = prediction_summary(test_lab,pred_lab,ntp,ntn,best_fs,best_epoch)
                    print('Test {0}'.format(mstr))

                    fs = mout['fscore']
                    if best_fs == 1.0 and test_histidx > 5 and test_hist[test_histidx-5:test_histidx].mean()==1.0:
                        print('Epoch %d: perfect fscore for the last %d iterations, no further training necessary'%(epoch,5))
                        break
                    elif fs > best_fs:
                        print('Epoch %d: new best fscore: %9.6f, saving model'%(epoch,100*fs))
                        best_epoch,best_fs = epoch,fs
                        best_weightf = train_weight_iterscoref%(best_epoch,unit2str(best_fs))
                        self.save_weights(best_weightf)
                        if save_preds:
                            best_predf = best_weightf.replace('weights','preds')
                            best_predf = splitext(best_predf)[0]+'.txt'
                            self.write_preds(best_predf,test_ids,test_lab,
                                             pred_lab,pred_out)
                        test_histidx = 0 # reset history since we found a new high score
                    elif test_histidx > stop_early and (best_fs-test_hist.max())>0:
                        msg = 'Epoch %d: no improvement for %d test epochs'%(epoch,stop_early)
                        if test_histdelta > 0 and best_fs < best_batch_fs:
                            msg += ', best_batch_fs (%9.6f) > best_fs (%9.6f), contining for history_delta (%d) additional epochs'%(best_batch_fs,best_fs,test_histdelta)
                            print(msg)
                        else:                                
                            msg += ', exiting training loop early'
                            print(msg)
                            break
                                                
                        test_hist[test_histidx%stop_early] = best_batch_fs
                        test_histidx = max(0,test_histidx-test_histdelta)
                        test_histdelta = 0
                    else:                
                        test_hist[test_histidx%stop_early] = fs
                        test_histidx += 1                    
                        
                    if plot_test_output:
                        # plot first layer activations for a few +/- examples/preds
                        fig = pl.figure(1)
                        pl.clf()
                        for i in [0,1]:
                            maski = tnmask if i==0 else tpmask
                            if plot_mispreds:
                                maski = maski & (test_lab!=pred_lab)
                            nmaski = min(nplotclass,maski.sum())
                            if nmaski > 0:
                                Xplotimg = self.transform(X_test[maski][:nmaski])
                                for j in range(nmaski):
                                    subplotidx = (i*nplotclass)+j+1
                                    pl.subplot(2, nplotclass, subplotidx)
                                    pl.imshow(Xplotimg[j,:],cmap='gray')
                                    pl.axis('off')
                        fig.canvas.draw()
                        pl.savefig(train_layer0_iterf%epoch)

                # if we already saved the model at best_iter, don't save it again
                if epoch % save_epoch == 0 and epoch != best_epoch:
                    out_weightf = train_weight_iterf%epoch
                    print('Epoch %d: saving model to file %s'%(epoch,out_weightf))
                    self.save_weights(out_weightf)
                    try:
                        if save_preds:
                            out_predf = out_weightf.replace('weights','preds')
                            out_predf = splitext(out_predf)[0]+'.txt'
                            self.write_preds(best_predf,test_ids,test_lab,
                                             pred_lab,pred_out)
                    except:
                        pass                

                print('Epoch %d: mean loss: %9.6f, processing time: %0.3f seconds'%(epoch,eloss/n_batches,etime))
                print('-'*60)

        except KeyboardInterrupt:
            if epoch > 0:
                print('Epoch %d: user interrupt, saving model before exiting'%epoch)            
                out_weightf = train_weight_iterf%epoch
                self.save_weights(out_weightf)
            pass        

        if epoch > 0:
            msg = 'Epoch %d: training stopped'%epoch
            if best_weightf:
                msg += '\n     best weights saved to %s'%best_weightf
            if out_weightf:
                msg += '\n     last set of weights saved to %s'%out_weightf
            print(msg)
            self.initialized = True
            
def update_base_outputs(base_model,**kwargs):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2 as activity_l2
    nb_hidden = kwargs.pop('nb_hidden',1024)
    nb_classes = kwargs.pop('nb_classes',2)
    obj_lambda2 = kwargs.pop('obj_lambda2',0.0025)

    print('Adding %d x %d relu hidden + softmax output layer'%(nb_hidden,
                                                               nb_classes))
    model = Sequential()
    model.add(base_model)
    model.add(Dense(nb_hidden, activation='relu'))
    model.add(Dense(nb_classes,activity_regularizer=activity_l2(obj_lambda2),
                    activation='softmax'))
    return model

def keras_model(model_base, model_flavor, **params):
    from keras import backend 
    from keras.optimizers import Adam
    verbose     = params.get('verbose',False)
    recompile   = params.pop('recompile',True)
    overwrite   = params.pop('overwrite',True)
    loss        = params.pop('loss','categorical_crossentropy')
    if recompile:
        print('Compiling base model')
        optimizer  = Adam(lr=update_lr, beta_1=beta_1, beta_2=beta_2,
                          epsilon=tol, decay=weight_decay)
        model_base.compile(loss=loss, optimizer=optimizer, **params)

    xform_func  = backend.function([model_base.layers[0].input,
                                    backend.learning_phase()],
                                   [model_base.layers[0].output])

    model_xform = lambda X: xform_func([X])[0]
    model_pred  = lambda X: model_base.predict(X,verbose=verbose)
    model_batch = lambda X,y: model_base.train_on_batch(X,y)
    model_save  = lambda outf: model_base.save_weights(outf,overwrite=overwrite)
    model_load  = lambda weightf: model_base.load_weights(weightf)

    return Model(package='keras',backend=backend.backend(),flavor=model_flavor,
                 base=model_base,batch=model_batch,transform=model_xform,
                 save=model_save,load=model_load,
                 predict=model_pred,params=params)

def lasagne_model(model_base, model_flavor, **params):
    import theano
    theano.config.floatX = 'float32'
    
    from theano import function as tfunction, shared as tshared
    from theano.tensor import tensor4, imatrix, nnet
    from theano.tensor import grad as Tgrad, mean as Tmean, reshape as Treshape

    from lasagne.utils import floatX
    from lasagne.updates import adam as lasagne_adam, total_norm_constraint
    from lasagne.layers import get_output as ll_output, \
        get_all_params as ll_all_params

    max_norm = 5.0
    
    verbose     = params.get('verbose',False)
    overwrite   = params.get('overwrite',True)

    sym_x = tensor4() # [nbatch,imgchan,imgrows,imgcols] dims
    sym_y = imatrix() # one-hot vector of [nb_class x 1] dims

    l_A_net      = model_base['A_net']
    l_transform  = model_base['transform']
    l_out        = model_base['net_out']        
    output_train = ll_output(l_out, sym_x, deterministic=False)
    output_shape = (-1, l_out.shape[1]) # nb_classes = l_out.shape[1]
    output_flat  = treshape(output_train, output_shape)
    output_loss  = nnet.categorical_crossentropy
    output_cost  = tmean(output_loss(output_flat+tol,sym_y.flatten()))

    trainable_params = ll_all_params(l_out, trainable=True)

    all_grads = tgrad(output_cost, trainable_params)
    updates, norm = total_norm_constraint(all_grads, max_norm=max_norm,
                                          return_norm=True)

    shared_lr = tshared(floatX(update_lr))
    updates = lasagne_adam(updates, trainable_params, learning_rate=shared_lr,
                           beta_1=beta_1, beta_2=beta_2, epsilon=tol)

    model_train = tfunction([sym_x, sym_y], [output_cost, output_train, norm],
                            updates=updates)

    output_eval, l_A_eval = ll_output([l_out, l_A_net], sym_x,
                                      deterministic=True)
    model_eval = tfunction([sym_x], [output_eval.reshape(output_shape),
                                     l_A_eval.reshape(output_shape)])
    model_batch = lambda X,y: model_train(X, int32(y))[0]
    model_pred  = lambda X: model_eval(X)[0]
    model_xform = lambda X: layer_output(X,l_transform)
    model_save  = lambda outf: save_all_weights(l_out,outf,overwrite=overwrite)
    model_load  = lambda weightf: load_all_weights(l_out,weightf)

    return Model(package='lasagne',backend='theano',flavor=model_flavor,
                 base=model_base,batch=model_batch,predict=model_pred,
                 transform=model_xform,save=model_save,load=model_load,
                 params=params)


class StateCallback(KerasCallback):
    def on_train_begin(self, logs={}):
        # called at the beginning of model training.
        self.losses = []

    def on_train_end(self, logs={}):
	# called at the end of model training.
        pass

    def on_epoch_begin(self, logs={}):
	# called at the beginning of every epoch.
        pass
        
    def on_epoch_end(self, logs={}):
	# called at the end of every epoch.
        pass
        
    def on_batch_begin(self, logs={}):
	# called at the beginning of every batch.
        pass
        
    def on_batch_end(self, logs={}):
	# called at the end of every batch.
        self.losses.append(logs.get('loss'))
        

def compile_model(input_shape,nb_classes,**params):
    package  = params.pop('model_package',default_model_package)
    flavor   = params.pop('model_flavor',default_model_flavor)    
    if package not in valid_model_packages:
        print('invalid model_package:',package)
        sys.exit(1)

    if flavor not in valid_model_flavors:
        print('invalid model_flavor:',flavor)
        sys.exit(1)
        
    if flavor == 'cnn3':
        if package=='keras':
            from models.cnn3_keras import model_init
        elif package=='lasagne':
            from models.cnn3_lasagne import model_init
    elif flavor == 'ffstn':
        params['ds_factor']=ds_ffstn
        if package=='keras':
            from ffstn_keras import model_init
        elif package=='lasagne':
            from ffstn_lasagne import model_init
    elif flavor == 'inceptionv3':
        from models.inceptionv3_keras import model_init
    elif flavor == 'xception':
        from models.xception_keras import model_init

    model_base = model_init(input_shape,**params)
    model_base = update_base_outputs(model_base)    

    if package=='keras':
        model = keras_model(model_base,flavor,**params)
    elif package == 'lasagne':
        model = lasagne_model(model_base,flavor,**params)

    if model.backend=='theano':
        model.transpose = (2,0,1) # channels first=(2,0,1)
    elif model.backend=='tensorflow':
        model.transpose = None # channels last=(0,1,2)=default
        
    print('Model+functions compiled for package',package,'flavor',flavor)

    return model

def apply_model(model,tile_dim,image_path,output_path,**kwargs):
    tile_path     = kwargs.pop('tile_path',None)
    prob_thresh   = kwargs.pop('prob_thresh',0.0)
    calc_salience = kwargs.pop('calc_salience',True)
    tile_stride   = kwargs.pop('tile_stride',None)
    hdr_path      = kwargs.pop('hdr_path',None)
    hdr_suffix    = kwargs.pop('hdr_suffix',None)
    verbose       = kwargs.pop('verbose',False)
    
    if isdir(image_path):
        image_files = glob(pathjoin(image_path,'*'+image_ext))
    else:
        image_files = [image_path]

    hdr_files = {}
    if hdr_path:
        for imagef in image_files:
            imgid = filename2flightid(imagef)
            hdrfiles = glob(pathjoin(hdr_path,imgid+'*'+hdr_suffix+'*.hdr'))
            msgtup=(imgid,hdr_path,hdr_suffix)
            if len(hdrfiles)==0:
                warn('no matching hdr for %s in %s with suffix %s'%msgtup)
                return            
            hdrf = hdrfiles[0]
            if len(hdrfiles)>1:
                msg = 'multiple .hdr files for %s in %s with suffix %s'%msgtup
                msg += '(using %s)'%hdrf
                warn(msg) 
            imgmap = mapinfo(openimg(hdrf.replace('.hdr',''),hdrf=hdrf),
                             astype=dict)
            imgmap['rotation'] = -imgmap['rotation']               
            hdr_files[imgid] = imgmap
                        
    for imagef in image_files:
        imgid = filename2flightid(imagef)
        imgmap = hdr_files.get(imgid,None)
        image_test = imread_image(imagef)
        if calc_salience:
            print('Computing salience map for image id {}'.format(imgid))
            salience_uls = collect_salience_uls(image_test,tile_dim,
                                                tile_stride=tile_stride)
            salience_tile_uls = gen_salience_tiles(image_test,salience_uls,
                                                   tile_dim)
            salience_out = predict_tiles(image_test,salience_tile_uls,
                                         len(salience_uls),verbose=verbose)
            
            output_prefix = '_'.join([imgid,'salience'])
            plot_pred_images(salience_out,mapinfo=imgmap,output_dir=output_dir,
                             output_prefix=output_prefix,mask_zero=False)
            
        if tile_path is None:
            continue
        
        tile_files = collect_tile_files(imgid,tile_path)
        if len(tile_files)!=0:
            file_tile_uls = gen_file_tiles(tile_files)
            pred_out = predict_tiles(image_test,file_tile_uls,len(tile_files),
                                     verbose=verbose)

            if plot_predictions:
                plot_pred_images(pred_out,mapinfo=imgmap,output_dir=output_dir,
                                 output_prefix=imgid,mask_zero=True)
            
            # geolocate detections with .hdr files
            if imgmap:
                from LatLongUTMconversion import UTMtoLL
                zone,hemi = imgmap['zone'],imgmap['hemi']
                zonealpha = zone + ('N' if hemi=='North' else 'M')
                tile_out = pred_out['tile_preds']
                keep_mask = float32(tile_out[:,-2])>=prob_thresh
                tile_keep = tile_out[keep_mask,:]
                tile_center = np.float32(tile_keep[:,[1,2]])+(tile_shape[0]//2)
                line,samp = tile_center.T
                utmx,utmy = sl2xy(samp,line,mapinfo=imgmap)
                lats,lons = UTMtoLL(23,utmy,utmx,zonealpha)
                preds,probs = tile_keep[:,-1],tile_keep[:,-2]
                csvf = pathjoin(output_dir,imgid+'_preds%d.csv'%int(prob_thresh))
                outcsv = []
                for i in range(tile_keep.shape[0]):
                    entryi = [imgid,'%d'%samp[i],'%d'%line[i],
                              '%18.4f'%utmx[i],'%18.4f'%utmy[i],zone,hemi,
                              lats[i],lons[i],preds[i],probs[i]]
                    outcsv.append(', '.join(map(lambda v: str(v).strip(),entryi)))
                with(open(csvf,'w')) as fid:
                    fid.write('\n'.join(outcsv)+'\n')
                print('saved',csvf)
