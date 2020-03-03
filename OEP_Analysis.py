# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:30 2020

@author: seany
"""

import scipy as sp
import numpy as np
import pyopenephys
import matplotlib.pyplot as plt

def readOEP_Bin(folder, exp_n=0,rec_n=0):
    file = pyopenephys.File(folder) 
    experiments = file.experiments    
    # recordings of first experiment
    experiment = experiments[exp_n]
    recordings = experiment.recordings

    # access first recording
    recording = recordings[rec_n]
    
    print('Duration: ', recording.duration)
    print('Sampling Rate: ', recording.sample_rate)
    
    analog_signals = recording.analog_signals
#    events_data = recording.events
    #spiketrains = recording.spiketrains
    # tracking_data are accessible only using binary format
 #   tracking_data = recording.tracking
    
    # plot analog signal of channel 4
    return analog_signals[0]

def signalplot(sig,xlim=(0,0,0)):
    fig_an, ax_an = plt.subplots(figsize=(10,20))
    if len(xlim)==2:
        ax_an.set_xlim(xlim)
    else:
        ax_an.set_xlim(np.array(sig.times).min(),np.array(sig.times).max())
    xloc=ax_an.get_xlim()[0]
    ax_an.spines['right'].set_visible(False)
    ax_an.spines['top'].set_visible(False)
    ax_an.spines['left'].set_visible(False)
    ax_an.xaxis.set_ticks_position('none')
    ax_an.xaxis.set_ticks_position('bottom')
    ax_an.set_yticks([])
    ax_an.set_xlabel('Time (s)')
    xsize=ax_an.get_xlim()[1]-ax_an.get_xlim()[0]
    for i,chan in enumerate(sig.signal):
        ax_an.plot(sig.times, chan+50000*i)
        print("Channel",i)
        ax_an.text(xloc-xsize/25,50000*i,'Channel '+str(i),ha='right')
    return fig_an,ax_an
        
#    return fig_an,ax_an