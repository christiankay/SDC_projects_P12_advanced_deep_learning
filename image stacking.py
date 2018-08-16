# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 18:19:32 2018

@author: knaak
"""

import numpy as np
import PIL
from PIL import Image
import imageio
import glob
images = []
filenames_1 =  glob.glob(r'C:\GIT\SDC_projects_P12_advanced_deep_learning\runs\1534433081.2969365_57_augmented\*.png')
filenames_2 =  glob.glob(r'C:\GIT\SDC_projects_P12_advanced_deep_learning\runs\1534421472.8349366_75_aumented\*.png')
filenames_3 =  glob.glob(r'C:\GIT\SDC_projects_P12_advanced_deep_learning\runs\1533591464.4407372_75ep_no_aug\*.png')

for name in range(len(filenames_1)):
    
    list_im = [filenames_1[name], filenames_2[name], filenames_3[name]]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    
    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( 'Trifecta.jpg' )    
    
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( 'Trifecta_vertical'+str(name)+'.jpg' )