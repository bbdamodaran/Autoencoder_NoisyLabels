# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:20:01 2018

@author: damodara
"""

import numpy as np
import dnn
from keras.callbacks import LearningRateScheduler
from architectures import mnist_featext, mnist_reconst
from architectures import cifar10_featext, cifar10_reconst
from architectures import labelpredictor



def get_model(data_set='mnist', main_input=None, nclass= None, classifier = True, drop_out=None,
              reconstruction=False, unsupervised_reconstruction=False):
    '''
    returns the keras model for the dataset
    when reconstruction is true, the model contains the decoder branch
    when classifier is false, the model contains on the feature extraction layer

    '''

    def L2_norm(vects):
        '''
        compute the squared L2 distance between two matrics
        '''
        x, y = vects
        ndim = x.shape
        ndim_y = y.shape
        if len(ndim) == 4:
            x = dnn.K.reshape(x, (-1, ndim[1] * ndim[2] * ndim[3]))
            y = dnn.K.reshape(y, (-1, ndim[1] * ndim[2] * ndim[3]))
        dist = dnn.K.reshape(dnn.K.sum(dnn.K.square(x), 1), (-1, 1))
        dist += dnn.K.reshape(dnn.K.sum(dnn.K.square(y), 1), (1, -1))
        dist -= 2.0 * dnn.K.dot(x, dnn.K.transpose(y))
        return dnn.K.sum(dist)

    def L2_norm_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], shape2[0])

    def switch_layer(vects):
        x,y, s_recon, us_recon = vects
        reconst = dnn.K.switch(dnn.K.greater_equal(x,y), us_recon, s_recon)
        return reconst

    if data_set == 'mnist':
        
        if reconstruction==False: 
            #fes = mnist_featext(main_input)
            fes_model = dnn.Model(main_input, mnist_featext(main_input), name='encoder') # feature ext model
            fes = fes_model(main_input)  
            if classifier:
                net = labelpredictor(fes, nclass)    # classifier                
                model = dnn.Model(main_input, net)
            else:
                model = fes_model
        else:
            fes, out_shape = mnist_featext(main_input, out_shape=True)
            fes_model = dnn.Model(main_input, fes, name='encoder')
            fes = fes_model(main_input)
            net = labelpredictor(fes, nclass, drop_out=drop_out)
            
            reconst = mnist_reconst(fes, out_shape)
            model = dnn.Model(main_input, [net, reconst])
            
        return model
        
    elif data_set == 'cifar10':
                    
        if reconstruction==False: 
            fes_model = dnn.Model(main_input, cifar10_featext(main_input), name='encoder') # feature ext model
            fes = fes_model(main_input)
            if classifier:
                net = labelpredictor(fes, nclass, drop_out=drop_out)
                model = dnn.Model(main_input, net)
            else:
                model = fes_model
        elif reconstruction == True and unsupervised_reconstruction ==False:
            fes, out_shape = cifar10_featext(main_input, out_shape=True)
            fes_model = dnn.Model(main_input, fes, name='encoder')
            fes = fes_model(main_input)
            
            net = labelpredictor(fes, nclass, drop_out=drop_out)
            reconst = cifar10_reconst(fes, out_shape)
            model = dnn.Model(main_input, [net, reconst])

        elif reconstruction == True and unsupervised_reconstruction == True:
            sup_fes, out_shape = cifar10_featext(main_input, out_shape=True)
            sup_fes_model = dnn.Model(main_input, sup_fes, name='sup_encoder')
            sup_fes = sup_fes_model(main_input)

            net = labelpredictor(sup_fes, nclass, drop_out=drop_out)
            sup_reconst = cifar10_reconst(sup_fes, out_shape)

            unsup_fes, out_shape = cifar10_featext(main_input, out_shape=True, name_prefix='unsup_')
            unsup_fes_model = dnn.Model(main_input, unsup_fes, name='unsup_encoder')
            unsup_fes = unsup_fes_model(main_input)

            unsup_reconst = cifar10_reconst(unsup_fes, out_shape, name_prefix='unsup_')


            stop_grad_sup = dnn.Lambda(lambda x: dnn.K.stop_gradient(x))(sup_reconst)
            stop_grad_unsup = dnn.Lambda(lambda x: dnn.K.stop_gradient(x))(unsup_reconst)
            total_reconst_unsup = dnn.keras.layers.add([unsup_reconst,stop_grad_sup])
            total_reconst_sup = dnn.keras.layers.add([stop_grad_unsup, sup_reconst])
            # #
            sup_diff = dnn.Lambda(L2_norm)([sup_reconst, main_input])
            unsup_diff = dnn.Lambda(L2_norm)([unsup_reconst, main_input])

            tot_recon = dnn.Lambda(switch_layer)([sup_diff, unsup_diff, total_reconst_sup, total_reconst_unsup])
            # #
            # total_reconst= dnn.K.switch(dnn.K.greater_equal(unsup_diff,sup_diff), total_reconst_unsup,  total_reconst_sup)
            # # total_reconst = dnn.Lambda(lambda x: x)(total_reconst)
            #con_cat = dnn.keras.layers.concatenate([sup_reconst, unsup_reconst], axis=0)
            model = dnn.Model(main_input, [net, tot_recon])

            
        return model
        


def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset: 
    :param noise_ratio:
    :return: 
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 120:
                return 0.0001
            elif epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)            
            
        