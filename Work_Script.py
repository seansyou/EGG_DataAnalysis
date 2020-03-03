# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:40:34 2020

@author: Sean
"""

import OEP_Analysis as oep


#folder='D:/Dropbox/open_ephys_recordings/2020-02-13_11-10-06'
#folder='D:/Dropbox/open_ephys_recordings/2020-02-13_11-23-15'
folder='D:/Dropbox/open_ephys_recordings/2020-02-13_11-37-45'



#dat2=oep.readOEP_Bin(folder,exp_n=1)
dat1=oep.readOEP_Bin(folder,exp_n=0)
#f,a=oep.signalplot(dat2,xlim=(51,54))
f,a=oep.signalplot(dat1,xlim=(250,300))
