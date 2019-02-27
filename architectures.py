# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:00:20 2018

@author: damodara
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:04:48 2017

@author: damodara
"""
import dnn



    
    
def svhnn_model(img_shape, n_class, l2_weight=0):
    model = dnn.Sequential()
    model.add(dnn.Convolution2D(64, (5, 5), input_shape=(img_shape[1], img_shape[2],img_shape[3]),
                                padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))
    model.add(dnn.MaxPooling2D(pool_size=(3,3), strides=2))
    # model.add(dnn.Dropout(0.9))
    model.add(dnn.Convolution2D(64, (5, 5), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))
    model.add(dnn.MaxPooling2D(pool_size=(3,3), strides=2))
    # model.add(dnn.Dropout(0.75))
    model.add(dnn.Convolution2D(128, (5,5), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))
    # model.add(dnn.Dropout(0.75)) 
    model.add(dnn.Flatten())
    model.add(dnn.Dense(3072,activation='relu',kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))
    # model.add(dnn.Dropout(0.25))
    model.add(dnn.Dense(2048,activation='relu',kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))
    # model.add(dnn.Dropout(0.25))
    model.add(dnn.Dense(n_class,activation='softmax',kernel_regularizer=dnn.keras.regularizers.l2(l2_weight)))

    return model
    
       


def mnist_feat_ext(main_input, l2_weight=0.0):
    net = dnn.Convolution2D(32, (5, 5), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(main_input)
    net = dnn.MaxPooling2D(pool_size=(2,2), strides=2)(net)  
    net = dnn.Convolution2D(48, (5,5),padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.MaxPooling2D(pool_size=(2,2),strides=2)(net)      
    net = dnn.Flatten()(net)
    net = dnn.Dense(100,activation='relu',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.Dense(100,activation='relu',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    return net
    
def assda_feat_ext(main_input, l2_weight=0.0):
    net = dnn.Convolution2D(32, (3, 3),padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(main_input)
    net = dnn.Convolution2D(32, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.MaxPooling2D(pool_size=(2, 2), strides=1)(net)
    
    net = dnn.Convolution2D(64, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.Convolution2D(64, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.MaxPooling2D(pool_size=(2, 2), strides=1)(net)
    
    net = dnn.Convolution2D(128, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.Convolution2D(128, (3, 3), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.MaxPooling2D(pool_size=(2, 2), strides=1)(net)
#   
    net = dnn.Flatten()(net)
    net = dnn.Dense(128,activation='sigmoid',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight),name='feat_ext')(net) 
#    net = dnn.Dense(512,activation='sigmoid',
#                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight),name='feat_ext')(net) 
    return net
    

    
    
def feat_ext(main_input, l2_weight=0.0):
    net = dnn.Convolution2D(32, (5, 5), padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(main_input)
    net = dnn.MaxPooling2D(pool_size=(2,2), strides=2)(net)  
    net = dnn.Convolution2D(48, (5,5),padding='same', activation='relu',
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.MaxPooling2D(pool_size=(2,2),strides=2)(net)      
    net = dnn.Flatten()(net)
    net = dnn.Dense(100,activation='relu',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    return net
    
# mnist fully connected
def mnist_fcn_featext(main_input, l2_weight=0.0):
    shape = main_input.shape.as_list()
    if len(shape)==4:
        net = dnn.Lambda(lambda x: dnn.keras.backend.reshape(x, (-1, shape[1]*shape[2]*shape[3])))(main_input)
    else:
        net = main_input
    
    net = dnn.Dense(500,activation='relu',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
#    net = dnn.Dropout(0.2)(net)
    net = dnn.Dense(500,activation='relu',
                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    
    return net

def classifier(model_input, nclass,l2_weight=0.0):
#    net = dnn.Dense(128,activation='relu',
#                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(model_input)
#    net = dnn.Dense(128,activation='relu',
#                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
#    net = dnn.Dropout(0.2)(model_input)
    net = dnn.Dense(nclass,activation='softmax', name='classifier_output')(model_input)
    return net
    
def classifier_dropout(model_input, nclass,l2_weight=0.0):
#    net = dnn.Dense(128,activation='relu',
#                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(model_input)
#    net = dnn.Dense(128,activation='relu',
#                        kernel_regularizer=dnn.keras.regularizers.l2(l2_weight))(net)
    net = dnn.Dropout(0.2)(model_input)
    net = dnn.Dense(nclass,activation='softmax', name='classifier_output')(net)
    return net
    
# MNIST
def mnist_featext(main_input, out_shape=False):
    net = dnn.Convolution2D(32, (3,3), activation='relu', padding='same')(main_input)
    net = dnn.MaxPooling2D((2,2),padding='same')(net)
    net = dnn.Convolution2D(64, (3,3), activation='relu',padding='same')(net)
    x1 = dnn.MaxPooling2D((2,2),padding='same')(net)
    net = dnn.Flatten()(x1)
    net = dnn.Dense(256, activation='relu', name= 'encoder')(net)
    if out_shape:
        return net, x1.get_shape().as_list()
    else:
        return net
                
def mnist_reconst(model_input, out_shape):           
    x = dnn.Dense(out_shape[1]*out_shape[2]*out_shape[3], activation='relu')(model_input)
    x = dnn.Reshape((out_shape[1], out_shape[2], out_shape[3]))(x)
    x = dnn.Convolution2D(64, (3,3), activation='relu',padding='same')(x)
    x = dnn.UpSampling2D((2,2))(x)
    x = dnn.Convolution2D(64, (3,3), activation='relu',padding='same')(x)
    x = dnn.UpSampling2D((2,2))(x)
    x= dnn.Convolution2D(1, (3,3), activation='linear',padding='same', name ='decoder')(x)
    return x
    
# CIFAR 10
def cifar10_featext(main_input, l2_weight=0.01, out_shape=False, name_prefix = 'sup_'):
    x = dnn.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block1_conv1')(main_input)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block1_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.MaxPooling2D((2, 2), strides=(2, 2), name=name_prefix+'block1_pool')(x)

    # Block 2
    x = dnn.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block2_conv1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block2_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.MaxPooling2D((2, 2), strides=(2, 2), name=name_prefix+'block2_pool')(x)

    # Block 3
    x = dnn.Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block3_conv1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'block3_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x1 = dnn.MaxPooling2D((2, 2), strides=(2, 2), name=name_prefix+'block3_pool')(x)

    x = dnn.Flatten(name=name_prefix+'flatten')(x1)

    x = dnn.Dense(256, kernel_initializer="he_normal", kernel_regularizer=dnn.l2(l2_weight), bias_regularizer=dnn.l2(l2_weight), name=name_prefix+'fc1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu', name=name_prefix+'encoder')(x)
    if out_shape ==False:
        return x
    else:
        return x, x1.get_shape().as_list()
        
def cifar10_reconst(model_input, out_shape, l2_weight=0.0, name_prefix = 'sup_'):
    x = dnn.Dense(out_shape[1]*out_shape[2]*out_shape[3])(model_input)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Reshape((out_shape[1], out_shape[2], out_shape[3]))(x)
    # block 3
    x = dnn.Convolution2D(196, (3, 3), padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight),name=name_prefix+'de_block3_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Convolution2D(196, (3, 3), padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight),name=name_prefix+'de_block3_conv1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.UpSampling2D((2,2),name=name_prefix+'block3_pool')(x)
    
    # block 2
    x = dnn.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'de_block2_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'de_block2_conv1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.UpSampling2D((2,2),name=name_prefix+'block2_pool')(x)
    
    # block 1
    x = dnn.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'de_block1_conv2')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name=name_prefix+'de_block1_conv1')(x)
    x = dnn.BatchNormalization()(x)
    x = dnn.Activation('relu')(x)
    x = dnn.UpSampling2D((2, 2),name=name_prefix+'block1_pool')(x)
    
    #
    x = dnn.Convolution2D(3, (3, 3), padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=dnn.keras.regularizers.l2(l2_weight), name =name_prefix+'decoder')(x)
    #x= dnn.BatchNormalization()(x)
    
    return x
    
def labelpredictor(model_input, nclass,l2_weight=0.0, drop_out=None):
    if drop_out is not None:
        model_input = dnn.Dropout(drop_out)(model_input)
    net = dnn.Dense(nclass,activation='softmax', name='classifier_output')(model_input)
    return net
    
