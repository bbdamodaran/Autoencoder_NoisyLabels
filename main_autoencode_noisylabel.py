# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:07:06 2018

@author: damodara

Autoencoders to avoid overfitting to the noisy lables: CCE + Sup_Autoencoder+ Unsup_Autoencoder
"""

import numpy as np
import matplotlib.pylab as plt
import dnn
import ot
import os
import matplotlib as mpl
#mpl.use('Agg')
#plt.switch_backend('agg')
from keras.utils.np_utils import to_categorical
from matplotlib.colors import ListedColormap

data_set = 'cifar10'

if data_set == 'mnist':
    from keras.datasets import mnist
    (traindata, trainlabel), (testdata, testlabel) = mnist.load_data()
    size = traindata.shape
    traindata = traindata/255.0
    testdata = testdata/255.0
    #traindata = traindata.reshape(-1, 28*28)
    #testdata = testdata.reshape(-1, 28*28)
    traindata = np.expand_dims(traindata, 3)
    testdata = np.expand_dims(testdata,3)
elif data_set == 'cifar10':
    from keras.datasets import cifar10
    (traindata, trainlabel), (testdata, testlabel) = cifar10.load_data()

    # from keras.datasets import mnist
    # (traindata, trainlabel), (testdata, testlabel) = mnist.load_data()
    # traindata = np.expand_dims(traindata,3)
    # testdata = np.expand_dims(testdata,3)

    from preprocess import zero_mean_unitvarince,resize_data
    size = traindata.shape
    traindata,mean,std = zero_mean_unitvarince(traindata,scaling=True)
    testdata,_,_ = zero_mean_unitvarince(testdata,scaling=True, channel_mean=mean, channel_std=std)

#%%

val_option = 'clean'
#%%
if val_option =='clean':
    from sklearn.model_selection import train_test_split
    cl_traindata, cl_val_data, cl_trainlabel, cl_val_label = train_test_split(traindata,
                                                trainlabel, test_size=0.1, random_state=42)
    cl_val_label_cat = to_categorical(cl_val_label)
#%%
pname = 'results/cifar10'
savefig = False
data_agument = True
reconstruction = False
#%% 
cl_trainlabel_cat = to_categorical(cl_trainlabel)
trainlabel_cat = to_categorical(trainlabel)
testlabel_cat = to_categorical(testlabel)
#%%
n_class = len(np.unique(trainlabel))
n_dim = np.shape(traindata)
#optim = dnn.keras.optimizers.Adam(lr=0.001)
optim = dnn.keras.optimizers.SGD(lr=0.01,momentum=0.9, decay=0, nesterov=False)

if data_agument:
    from keras.preprocessing.image import ImageDataGenerator
    # datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
    #                              height_shift_range=0.2,
    #                              horizontal_flip=True)
    #
    datagen = ImageDataGenerator(width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)

#%%
def build_filename(noise, alpha, lamda, reg):
    return ('org_jdot_model'+ '_' +'n_' +str(noise) + '_' +'al_'+str(alpha)+'_'
                +'lam_'+str(lamda)+'_'+'reg_'+str(reg)+'.hd5')
#%% clean model
from architectures import  classifier_dropout, assda_feat_ext,classifier
if len(n_dim)==4: 
    ms = dnn.Input(shape=(n_dim[1],n_dim[2], n_dim[3]))
else:               
    ms = dnn.Input(shape=(n_dim[1],))
from models import get_model
if reconstruction == False:
    clean_model = get_model(data_set, ms, nclass=n_class, classifier=True, drop_out=0.5)
    clean_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    clean_model = get_model(data_set, ms, nclass=n_class, classifier=True)
    clean_model.compile(optimizer=optim, loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy'])

epochs=1
cfn='clean_model_weights1.hd5'
early_stop = dnn.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=25, verbose=1,
                                                   mode='auto')
saveBestModel = dnn.keras.callbacks.ModelCheckpoint(cfn, monitor='val_loss', verbose=1,
                                                         save_best_only=False,
                                                         mode='auto')
if data_agument:
    clean_model.fit_generator(datagen.flow(cl_traindata, cl_trainlabel_cat, batch_size=128),
                              steps_per_epoch=len(traindata) / 128,
                              epochs=epochs, verbose=1, validation_data = (cl_val_data, cl_val_label_cat),
                              callbacks=[early_stop,saveBestModel])
else:
    clean_model.fit(traindata, trainlabel_cat, batch_size=128, epochs=epochs, validation_split=0.1,
                    callbacks=[early_stop,saveBestModel])

clean_model.load_weights('clean_model_weights.hd5')

clean_trainacc = clean_model.evaluate(cl_traindata, cl_trainlabel_cat)
clean_testacc = clean_model.evaluate(testdata, testlabel_cat)
print("clean model train acc", clean_trainacc)
print("clean model test acc", clean_testacc)

pretrained_weights = clean_model.layers[1].get_weights()

#%%
noise=np.array([0.4])
#noise=np.arange(0,1,0.1)
crossent_ntrain_acc = np.zeros((len(noise),1))
crossent_train_acc = np.zeros((len(noise),1))
crossent_test_acc = np.zeros((len(noise),1))

val_option ='noisy'
#%%
#fname = 'mnist_selfw_acc.txt'
#f_txt = fopen(os.path.join(pathname, fname), 'wb')
#%% Noisy lables
asymmetric =0
for i in np.arange(len(noise)):
    from simulate_noisylabel import noisify_with_P,mnist_simulate_noisylabel, noisify_cifar10_asymmetric
    if noise[i]>0.0:
        if asymmetric==1:
            n_trainlabelf = noisify_with_P(trainlabel.ravel(), noise=noise[i])
        elif asymmetric==0:
            if data_set == 'mnist':
                n_trainlabelf = mnist_simulate_noisylabel(trainlabel, noise=noise[i])
            elif data_set == 'cifar10':
                n_trainlabelf,_ = noisify_cifar10_asymmetric(trainlabel.ravel(), noise=noise[i])
    else:
        n_trainlabelf = trainlabel.copy()

    if val_option=='noisy':
        from sklearn.model_selection import train_test_split
        n_traindata, val_data, n_trainlabel, val_label = train_test_split(traindata,
                                                n_trainlabelf, test_size=0.1, random_state=42)
        tn_traindata, tval_data, tn_trainlabel, tval_label = train_test_split(traindata,
                                                                          trainlabel, test_size=0.1, random_state=42)
        val_label_cat = to_categorical(val_label)
    n_trainlabel_cat = to_categorical(n_trainlabel)
     
#%% noisy model
    #fn = 'bestweights_asym_0.4.hd5' % best weights for the asym 0.4
    fn ='dummyweights.hd5'

    optim = dnn.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False, decay=0.0001)
    #optim = dnn.keras.optimizers.Adam(lr=0.001)
    ms1 = dnn.Input(shape=(n_dim[1],n_dim[2],n_dim[3]))
    early_stop = dnn.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=25, verbose=1,
                                                   mode='auto')
    saveBestModel6 = dnn.keras.callbacks.ModelCheckpoint(fn, monitor='val_loss', verbose=1,
                                                         save_best_only=True,
                                                         mode='auto')
    epochs = 120

    def scheduler(epoch):
        #            print("coral param to {}".format(lr))
        dnn.K.set_value(re_param, (epoch + 1)/epochs)
        dnn.K.set_value(cce_param, (1.-(1/epochs)*epoch))
        #            print("l am changed to {}".format(dnn.K.get_value(self.coral_param)))
        return dnn.K.get_value(noisy_model.optimizer.lr)

    def lr_scheduler(epoch):
        if epoch > 80:
            return 0.001
        elif epoch > 40:
            return 0.01
        else:
            return 0.01


    callback_re, lr_callback = dnn.LearningRateScheduler(scheduler), dnn.LearningRateScheduler(lr_scheduler)

    cce_param = dnn.K.variable(1.0)
    re_param = dnn.K.variable(0.2)

    def L2_norm(vects):
        '''
        compute the squared L2 distance between two matrics
        '''
        x, y = vects
        ndim = x.shape
        if len(ndim) == 4:
            x = dnn.K.reshape(x, (128, ndim[1] * ndim[2] * ndim[3]))
            y = dnn.K.reshape(y, (128, ndim[1] * ndim[2] * ndim[3]))
        dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(x), 1), (-1, 1))
        dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(y), 1), (1, -1))
        dist -= 2.0 * dnn.K.dot(x, dnn.K.transpose(y))
        return dnn.K.sum(dist)


    batch_size = 128
    def reconst_loss(y_true, y_pred):
        sup_reconst = y_pred[:batch_size]
        unsup_reconst = y_pred[batch_size:]
        sup_diff = L2_norm((sup_reconst, y_true))
        unsup_diff = L2_norm((unsup_reconst, y_true))
        # total_reconst_unsup = unsup_reconst+dnn.K.stop_gradient(sup_reconst)
        # total_reconst_sup = dnn.K.stop_gradient(unsup_reconst) +sup_reconst
        total_reconst_unsup = unsup_reconst
        total_reconst_sup =  sup_reconst
        reconst = dnn.K.switch(dnn.K.greater_equal(unsup_diff,sup_diff), total_reconst_unsup,  total_reconst_sup)
        loss = dnn.K.binary_crossentropy(y_true, reconst)
        return loss



    if reconstruction == False:
        noisy_model = get_model(data_set, ms1, nclass=n_class, classifier=True, drop_out=0.5)
        noisy_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

        if data_agument:
            noisy_model.fit_generator(datagen.flow(n_traindata, n_trainlabel_cat, batch_size=128),
                                      steps_per_epoch=len(n_traindata) / 128,
                                      epochs=epochs, verbose=1, validation_data=(n_traindata, tn_trainlabel_cat),
                                      callbacks=[early_stop,saveBestModel6, lr_callback])
        else:
            noisy_model.fit(n_traindata, n_trainlabel_cat, batch_size=128, epochs=epochs, verbose=1,
                            validation_data=(val_data, val_label_cat), callbacks=[early_stop, saveBestModel6])

    else:
        noisy_model = get_model(data_set, ms1, nclass=n_class, classifier=True, reconstruction = True,
                                unsupervised_reconstruction = True, drop_out=0.0)
        noisy_model.compile(optimizer=optim, loss=['categorical_crossentropy', 'mse'],
                            metrics=['accuracy'], loss_weights=[cce_param, re_param])
        if data_agument:
            noisy_model.fit_generator(datagen.flow(n_traindata, n_trainlabel_cat, batch_size=128),
                                      steps_per_epoch=len(n_traindata) / 128,
                                      epochs=epochs, verbose=1, validation_data=(val_data, val_label_cat),
                                      callbacks=[early_stop])
        else:
            noisy_model.fit(n_traindata, [n_trainlabel_cat, n_traindata], batch_size=128, epochs=epochs, verbose=1,
                            validation_data=(testdata, [testlabel_cat, testdata]))





    noisy_model.load_weights('bestweights_asym_0.4.hd5')
    #%%
    crossent_ntrain_acc[i] = noisy_model.evaluate(n_traindata, n_trainlabel_cat)[1]
    crossent_ttrain_acc = noisy_model.evaluate(n_traindata, to_categorical(tn_trainlabel))[1]
    crossent_train_acc[i] = noisy_model.evaluate(traindata, trainlabel_cat)[1]
    crossent_test_acc[i]= noisy_model.evaluate(testdata, testlabel_cat)[1]
    print("crossent ntrain acc", crossent_ntrain_acc[i])
    print("crossent train acc", crossent_train_acc[i])
    print("crossent test acc", crossent_test_acc[i])
#    noisy_train_softmax = noisy_model.predict(n_traindata)
#    noisy_entropy = np.sum(-noisy_train_softmax*np.log(noisy_train_softmax),1)