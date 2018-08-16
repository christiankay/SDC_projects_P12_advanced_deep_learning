# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 00:43:30 2018

@author: knaak
"""

import imageio
import glob
images = []
filenames =  glob.glob(r'C:\GIT\SDC_projects_P12_advanced_deep_learning\joined_gif\*.jpg')
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(r'C:\GIT\SDC_projects_P12_advanced_deep_learning\joined_gif_13-60.gif', images, format='GIF', duration=1)