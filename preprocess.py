# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:05:29 2017

@author: damodara
"""

import numpy as np




def zero_mean_unitvarince(data, scaling=False, channel_mean=None, channel_std=None):
    size = data.shape
    data = np.float64(data) 
    if len(size)==2: # for the 2D data
        from sklearn.preprocessing import scale
        mean_data = np.mean(data,axis=0)
        std_data = np.std(data, axis=0)
        data = data - np.tile(mean_data, [size[0], 1])
        data = data/(np.tile(std_data, [size[0],1]))
    elif len(size)>2:# for the images
        mean_image = np.mean(data, axis=0)
        std_image = np.std(data, axis=0)
        mean_size = mean_image.shape
        if channel_mean is None:
            channel_mean = np.mean(mean_image, axis=(0,1))
            channel_std = np.std(data, axis=(0,1,2))
        if len(mean_size)>2:
            # per channel mean subtraction
           for i in range(mean_size[2]):
               data[:,:,:,i] = data[:,:,:,i]-channel_mean[i]
               if scaling:
                   data[:,:,:,i] = data[:,:,:,i]/(channel_std[i]+1e-5)
        else:
            data = data-channel_mean
            if scaling:
                data = data/(channel_std+1e-5)
                
    data[np.isnan(data)]=0       
    return data, channel_mean, channel_std
    
def resize_data(data, resize_size):
     
     from scipy.misc import imresize
     data_type = type(data)
     
     if data_type==list:
         s= data[0].shape
         ndata= len(data)
     else:
         s=data[0].shape
         ndata = data.shape[0]
     
     if len(s)==2:
         data1=np.zeros((ndata,resize_size,resize_size))
         
         for i in range(ndata):
             data1[i] = imresize(data[i], (resize_size,resize_size))
     elif len(s)==3:
         data1=np.zeros((ndata,resize_size,resize_size,3))
         
         for i in range(ndata):
             data1[i] = imresize(data[i], (resize_size,resize_size))
     del data         
     return data1
#    
#    if (len(size)==4):
#        for i in range(size[0]):
#            for j in range(size[3]):
#                data[i,:,:,j]= data[i,:,:,j]-np.mean(data[i,:,:,j])
#    elif (len(size)==3):
#        for i in range(size[0]):
#            data[i,:,:]= data[i,:,:]-np.mean(data[i,:,:])
#            
#    if min_max_scale:
#       from sklearn.preprocessing import minmax_scale
#        
#       if (len(size)==4):
#           for i in range(size[3]):
#                tmp = minmax_scale(data[:,:,:,i].reshape(-1, size[1]*size[2]), axis=1)
#                data[:,:,:,i] = tmp.reshape(-1,size[1],size[2])
#       elif (len(size)==3):
#            data = minmax_scale(data.reshape(-1, size[1]*size[2]), axis=1)
#            data = data.reshape(-1, size[1],size[2])               
#                
#
#    return data            

                
 
  