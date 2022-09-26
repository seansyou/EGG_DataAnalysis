# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:42:42 2021

@author: seany
"""
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import interp1d
from matplotlib.offsetbox import AnchoredOffsetbox

def read_egg_v2(file,channels=1,header=7,rate=32):
    """
    Data import for EGGv2. 
    
    Parameters
    ----------
    file : string
        filepath to data from EGG recording.
    channels : int, optional
        Number of channels, used for parsing data. The default is 1.
    header : int, optional
        Number of leading lines to skip simple transmission msgs. The default is 7.
    rate : float, optional
        sampling rate off measurement. The default is 32.

    Returns
    -------
    datlist : list of 2xN numpy arrays 
        Each array indicates one channel of the recording, where [0,:] are timestamps and [1,:] are measurements
    """
    dat=pd.read_csv(file,header=header)
    datarray=[]
    #samples per second
    #rate=32
    #rate=8
    dup_i=0
    for row in range(len(dat)):
        duparray=np.array([False])
        #Remove duplicates in array from ack failure during data transmission
        if row > 0:
            duparray=np.array(dat.iloc[row,1:-1])==np.array(dat.iloc[row-1,1:-1]) 
        if all(duparray): 
            dup_i+=1
        else:
    #        print(row)
            for column in range(len(dat.iloc[row])):            
                if column == 0:
                    element=dat.iloc[row,column]
                    mod=element.split('>')
                    datarray.append(int(mod[1]))
                if column > 0 and column < 30:
                    datarray.append(int(dat.iloc[row,column]))
    datarray=np.array(datarray)
    convarray=[]
    timearray=np.array(np.arange(len(datarray))/rate)
    for ele in datarray:
        if ele<2**15:
            convarray.append(0.256*ele/(2**15))
        else:
            convarray.append(0.256*(((ele-2**16)/(2**15))))
    voltarray=np.array(convarray)
    voltarray.shape
    timearray.shape
    size=np.int(voltarray.size/channels)
    print(size)
    reshaped_volt=np.reshape(voltarray,(size,channels))
    reshaped_time=np.reshape(timearray,(size,channels))
    
    datlist=[]
    for num in range(channels):
        datlist.append(np.array([reshaped_time[:,num],reshaped_volt[:,num]*1000])) #have to convert to mV
    return datlist

def egg_interpolate(dat,rate=62.5,start_value=0,end_value=0):
    f=interp1d(dat[0,:],dat[1,:])
    if start_value==0: start_value=dat[0,:].min()
    if end_value==0:end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    return tfixed, f(tfixed)

def egg_filter(dat,rate=32,freq=[0,0.1],order=3):
    """
    Function which filters data using a butterworth filter
    Parameters
    ----------
    dat : List of 2 np arrays
        List of 2 np arrays where first array are timestamps and 2nd array is values
    rate : sampling rate in seconds, optional
        Sampling rate in seconds, used for interpolation of data prior to filtering. The default is 32.
    freq : List, optional
        Bandpass filter frequency. The default is [0,0.1].
    order : int, optional
        Order of butterworth filter generated for filtering. The default is 3.
    Returns
    -------
    fdata: numpy array of 2xN.
        1st index is columns, 2nd is rows. 1st column are timestamps and 2nd column is filtered data.

    """
    fn=rate/2
    wn=np.array(freq)/fn
#    wn[0]=np.max([0,wn[0]])
    wn[1]=np.min([.99,wn[1]])
    print(wn)
    f=interp1d(dat[0,:],dat[1,:])
    start_value=dat[0,:].min()
    end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    b,a=sig.butter(order,wn,btype='bandpass')
    filtered=sig.filtfilt(b,a,f(tfixed))
    fdata=np.array([tfixed,filtered])
    return fdata

def egg_fft(dat,rate=32,xlim=[-5,5],ylim=[0,.25]):
    f=interp1d(dat[0,:],dat[1,:])
    start_value=dat[0,:].min()
    end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    fftdat=fftpack.fft(f(tfixed))
    freqs=fftpack.fftfreq(len(f(tfixed)))*rate*60
    fig, ax = plt.subplots()
    ax.stem(freqs, np.abs(fftdat))
    ax.set_xlabel('Frequency in 1/mins')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(xlim)
#    ax.set_ylim(-5, 11000)
    return freqs,fftdat

def egg_powerspectrum(dat, rate=62.5,vlines=[]):
    x,y=egg_interpolate(dat,rate=rate)
    f, pden=sig.periodogram(y,fs=rate)
    figP=plt.figure()
    ax_P=figP.add_subplot(111)
    ax_P.loglog(f, pden)
    ax_P.set_ylim([1e-7,1e6])
    ax_P.set_xlim([.001,20])
    ax_P.set_ylabel('Power')
    ax_P.set_xlabel('Frequency (Hz)')
    ax_P.vlines(vlines,ymin=0,ymax=1e10,linewidth=1,color='black')
    return figP, ax_P

def read_egg_v3(file,header=0,rate=62.5,scale=150,error=0):
    """
    This is a function which uses pandas to read in data recorded from EGG V3 and transmitted to a board using
    RFStudio7. 
    
    file : filepath of the target txt file
    header : Number of lines to skip
    rate : Sampling rate in samples/second per channel set on the ADS131m8
    scale : +- scale in mV 
    error : returns data with CRC errors. Default is 0 so those are stripped
    
    output: Pandas data frame with the following information:
        .realtime : realtime from RFStudio when packet was received
        .misc : RF Studio output, not useful
        .packet : packet number, set from EGGv3, ranges from 0 to 65535 (unit16). Roll over if higher
        .msg : str of packet recieved
        .rssi : RSSI of packet, also includes CRC error
        'Channel n': Channels of recording data in mV, n is from 0 to 7
        .counter : absolute renumbered packets (without overflow)
        .timestamps : timesamples calculated from sampling rate and absolute timer
        .SPI : SPI Status (first packet of msg)
    
    """
    dat=pd.read_csv(file, header=header, dtype = str, delimiter='|', names=['realtime','misc','packet','msg','rssi'])
    dat=dat[~dat.rssi.str.contains('error')]
    dat=dat[dat.misc.str.contains('16')]
    dat=dat.reset_index(drop=True)
    dat_col=dat.msg
    hexdat=dat_col.str.split(' ') #Return list of splits based on spaces in msg
    serieslist=[]
    for k,ele in enumerate(hexdat):
        if len(ele) == 23: #Only select those that have the correct length
            vlist=[]
            for i in range(0,10):
                n=i*2+2
                value= ''.join(['0x',ele[n],ele[n-1]])
                hvalue=int(value,16)
                if i==0:
                    vlist.append(hvalue) #append hex code
                else:    
                    if hvalue<2**15:
                        vlist.append(scale*float(hvalue)/(2**15))
                    else:
                        vlist.append(scale*(((float(hvalue)-2**16)/(2**15))))
        else:
#            print('Line Error!'+str(k))
#            print(ele)
            vlist=[] #add empty list on error
        serieslist.append(vlist)
    collist=['SPI']
    for i in range(8): collist.append('Channel '+str(i)) #make channel list name
    collist.append('CRC')
    datalist=pd.DataFrame(serieslist,columns=collist)
    print(datalist)
    print(dat)
    fulldat=pd.concat((dat,datalist),axis=1)
    print(fulldat)
    counter=fulldat.packet.astype(int)
    new_counter=[0]
    for j,ele in enumerate(counter[1:]): #Renumbered counter - note this will give an error if you accidentally miss the 0/65535 packets
        step=counter[j+1]-counter[j]
#       if step != -65535:
        if step > 0:
            new_counter.append(step+new_counter[j])
#       elif step < 0:
#            new_counter.append(new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
#            new_counter.append(1+new_counter[j])
    tarray=np.array(new_counter)*1/62.5
    abscounterseries=pd.Series(new_counter,name='counter')
    tseries=pd.Series(tarray,name='timestamps')
    
    fulldat=pd.concat((fulldat,abscounterseries,tseries),axis=1)
    noerror=~fulldat.rssi.str.contains('error') # Gives rows without crc error
    if error: 
        return fulldat # return non-crc error 
    else:
        return fulldat[noerror]
#    hexdat.dropna() #drop out of range NaNs without shifting indicies
    
def signalplot(dat,xlim=(0,0,0),spacer=0,vline=[],freq=1,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,20),textsize=16,hline=[]):
    """
    Function to plot all channels in dataframe following data import using read_egg_v3
    
    Inputs:
        sig: Dataframe containing time data in "timestamps" column and "Channel n" where n is channel number
        xlim: list of 2 elements taking time range of interest. Default behavior is to full timescale
        spacer: Spacing between plots, and their scaling. Default behavior is spaced on max y value in a channel
        freq: frequency list in Hz, 2 element list, for bandpass filtering. Default is no filtering
        order: order of butter filter used in filtering
        rate: sampling rate of data, for filtering
        title: title label of plot, default is none 
        vline: list of float marking lines for xvalues, usually for fast visulation/measurement
        skip_chan: list of channels to skip, default none. 
        figsize: tuple of figure size dimensions passed to matplotlib.figure, default 10,20
        
    Outputs:
        fig_an: figure instance of plot (from matplotlib)
        ax_an: axis instance of plot
        Outarray.T: exported filtered data
    """
    x=dat.timestamps.to_numpy()
    outarray=[]
    if freq==1: outarray.append(x)
    plt.rcParams['font.size']=textsize
    fig_an, ax_an = plt.subplots(figsize=figsize)
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]
    xloc=ax_an.get_xlim()[0]
    ax_an.spines['right'].set_visible(False)
    ax_an.spines['top'].set_visible(False)
    ax_an.spines['left'].set_visible(False)
    ax_an.xaxis.set_ticks_position('none')
    ax_an.xaxis.set_ticks_position('bottom')
    ax_an.set_yticks([])
    ax_an.set_xlabel('Time (s)')
    xsize=ax_an.get_xlim()[1]-ax_an.get_xlim()[0]   

    loc=np.logical_and(x>xlim[0],x<xlim[1])
    space=0
    if spacer == 0: #this is to automatically set the spacing we want between the 
        distarr=[]
        for i,column in enumerate(dat.columns):
            if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
                y=dat[column].to_numpy()                
                if freq == 1:
                    distance=y[loc].max()-y[loc].min()
                else:
                    mod=egg_filter(np.array([x,y]),freq=freq,rate=rate,order=order)
                    loc2=np.logical_and(mod[0,:]>xlim[0],mod[0,:]<xlim[1])
                    distance=mod[1,loc2].max()-mod[1,loc2].min()
                
                distarr.append(distance)
        distarr=np.array(distarr)
        print(distarr)
        spacer=distarr.max()*1.1    

    for i,column in enumerate(dat.columns):
        if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
            y=dat[column].to_numpy()
            
            if freq == 1:
                ax_an.plot(x, y-y[loc].mean()+space)
                print('plotted!')
                outarray.append(y)
            else:
                mod=egg_filter(np.array([x,y]),freq=freq,rate=rate,order=order)
                if len(outarray)==0: outarray.append(mod[0,:].squeeze())
                ax_an.plot(mod[0,:], mod[1,:]+space)
                outarray.append(mod[1,:].squeeze())
            print(dat[column].name)
            ax_an.text(ax_an.get_xlim()[0]-xsize/40,space,dat[column].name,ha='right')
            space+=spacer
            print(space)
    if len(vline) != 0:
        ax_an.vlines(vline,ymin=0-spacer/2, ymax=space-spacer/2,linewidth=5,color='black',linestyle='dashed')
    if len(hline) != 0:
        ax_an.hlines(hline,xmin=xlim[0],xmax=xlim[1],linewidth=5,color='black',linestyle='dashed')
    ax_an.set_ylim(0-spacer,space)
    ax_an.set_title(title)
    ax_an.vlines(xlim[0],ymin=0-3*spacer/4,ymax=0-spacer/2,linewidth=10,color='black')
    ax_an.text(xlim[0]+xsize/40,0-5/8*spacer,str(np.round(spacer*1/4,decimals=2))+' mV',ha='left')
#    add_scalebar(ax_an,hidex=False,matchy=True)
    outarray=np.array(outarray)
    loc_out=np.logical_and(outarray[0,:]>xlim[0],outarray[0,:]< xlim[1])
    outarray=outarray[:,loc_out]
    return fig_an,ax_an,outarray.T

def heatplot(dat,xlim=(0,0,0),spacer=0,vline=[],freq=1,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,0,0],interpolation='bilinear',norm=True):
    plt.rcParams['font.size']=textsize
    fig_an, ax_an = plt.subplots(figsize=figsize)
    x=dat.timestamps.to_numpy()
    arraylist=[]
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]
        
        
    for i,column in enumerate(dat.columns):
        if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
            y=dat[column].to_numpy()
            if freq == 1:
                xf,yf=egg_interpolate(np.array([x,y]),rate=rate)
            else:
                d=np.array([x,y])
                mod=egg_filter(d,freq=freq,rate=rate,order=order)
                xf=mod[0,:]
                yf=mod[1,:]
            arraylist.append(yf)
            
    datlist=np.array(arraylist)
    if len(xlim) == 2:
        loc2=np.logical_and(xf>xlim[0],xf<xlim[1])
        datlist=datlist[:,loc2]
    if norm == True:
        datlist=np.absolute(datlist)
    if len(vrange)==2:
        colors=ax_an.imshow(np.flip(datlist,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',vmin=vrange[0],vmax=vrange[1],interpolation=interpolation)
    else:
        colors=ax_an.imshow(np.flip(datlist,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',interpolation=interpolation)  
    ax_an.set_xlabel('Time (s)')
    ax_an.set_ylabel('Channel Number')    
    cbar=fig_an.colorbar(colors,ax=ax_an)
    cbar.set_label('Electrical Activity (mV)', labelpad=10)
    return fig_an,ax_an,datlist



def rssiplot(dat,xlim=[0,0,0],figsize=(5,5),ylim=[-100,-20],textsize=16):
    plt.rcParams['font.size']=textsize
    x=dat.timestamps.to_numpy()
    y=np.asarray(dat.rssi.to_numpy(),dtype=np.float64)
    fig_an, ax_an = plt.subplots(figsize=figsize)
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]            
    ax_an.set_ylim(ylim)
#    ax_an.set_yticks([-100,-80,-60,-40,-20])
    ax_an.plot(x,y,'ro',markersize=.5)
    ax_an.set_ylabel("RSSI (dB)")
    ax_an.set_xlabel("Time (s)")
    return fig_an,ax_an

#########PEAK TRACKING LIBRARIES BETA

def time_and_HR (array_filepath, seg_length = 30, time='mean', plot='yes',peak='yes', thres=0.5):
    """
    Function to plot the Time vs. HR from data in  exported by signal plot (2 column)
    
    Inputs:
        array_filepath: import your array.txt in here
        seg_length: The number of seconds you want each peak detection graph to be
        time: 3 options: 'max, 'min', 'mean'. 'max' sets the largest time value in each array for each HR, 'min' takes the smallest, and 'mean' takes the average. 
        plot: option to plot time vs. HR graph. plot='yes' to plot, plot= anything else to not plot
        peak: option to plot the peak graphs. plot='yes' to plot, plot= anything else to not plot
        thres:(float between [0., 1.]) – Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        
    Outputs:
        blank_array: an array with all data values of Time vs. HR
    """
    array= np.genfromtxt(array_filepath)
    x_axis=array[:,0]
    y_axis=array[:,1]*-1
    num_of_xsplits=round((x_axis.max()-x_axis.min())/seg_length) #formula for splitting entire time into 20 segments of same length
    
    splitx= np.array_split(x_axis, num_of_xsplits)  #actually splitting up the time
    splity= np.array_split(y_axis, num_of_xsplits) # splitting up the voltage as well
    
    
    blank_array= np.zeros([num_of_xsplits,2], dtype=float)  #creating an array of zeros on 2 columns
    for i,arr in enumerate(splitx):
        new_indexes = peakutils.indexes(splity[i], thres, min_dist=20) #using peak_detection library to find moments of peaks
        total_time = splitx[i].max()-splitx[i].min()    
        HR = 60*len(new_indexes)/total_time                #calculating HR by dividing total number of peaks by total time
        blank_array[i,1]=HR
        if time=='mean': blank_array[i,0]=arr.mean()
        if time=='min' : blank_array[i,0]=arr.min()
        if time=='max' : blank_array[i,0]=arr.max()
        if peak=='yes':### I put this into the for loop so it plots all of the segments - Sean
            pyplot.figure(figsize=(10,6))
            pplot(splitx[i], splity[i], new_indexes)
    if plot=='yes':
        fig=pyplot.figure()
        ax=fig.add_subplot(111)        
        ax.plot(blank_array[:,0],blank_array[:,1])
    print('++++++++++++++++++++++++++++')
    print(blank_array)
    return blank_array




def import_PR (CSV_filepath, droplist, offset=0):
    """
    Function to plot the Time vs. PR from data in CSV file generated by animal monitor
    
    Inputs:
        CSV_filepath: filepath of CSV file
        droplist: list of rows that you want to drop.
        offset: the amount of time offset to match the time_and_HR graph
        
    Outputs:
        blank_array: an array with all data values of Time vs. PR
    """
    df= pd.read_csv(CSV_filepath, skiprows=3,usecols=[0,1,2,3,4]) #skipping some rows and using only first 4 columns
    df2=df.drop(droplist)                                #getting rid of data to match starting point w/ HR data
    df3=df2.drop(df2[df2['PR']=='--'].index) 
        
    index =range(len(df3)) # creating a new index range bc of all the df.drops
    df3.index = index
    df3['Time']=df3['Time'].str[:-3] #getting rid of the :00 seconds from all times to make easier calculations
    hours_to_sec=df3['Time'].str.split(':', n=1).str[0].astype('float')*3600 #splitting up the times by ':' and then turning hours into seconds
    min_to_sec= df3['Time'].str.split(':', n=1).str[1].astype('float')*60 #splitting up times and then turning minutes into seconds
    total_sec= (hours_to_sec + min_to_sec)-offset #adding seconds together and offsetting to match w/ HR datapoints       
    x=total_sec
    y=df3['PR'].astype("float")
    
    array=np.column_stack((x,y))  #combines both columns
    return array
    



def plot_peak_freq(filepath,seg_length, save_fig_location='', thres=.5,xlim=[0,100000],freq=[.02,.2],min_dist=20):
    """
    Function to plot graph of all channels frequencies
    
    Inputs:
        CSV_filepath: filepath of CSV file
        seg_length: the size of the pieces of data that the peak detection will run on, in seconds. 
        save_fig_location: filepath to store all the peak detection graphs
        thres:(float between [0., 1.]) – Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        xlim: list of 2 elements taking time range of interest. Default behavior is to full timescale
        freq: frequency list in Hz, 2 element list, for bandpass filtering. Default is no filtering
        min_dist: (int) Minimum distance in number of array elements between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint.
    Outputs:
        output_arr: array of time vs. all channels' HR
        axs: plots frequencies of all channels in separately
        ax: plots frequenices of all channels in single graph
    """
    dat= read_egg_v3(filepath,header=0,rate=62.5,scale=150,error=0)
    x=signalplot(dat,xlim=xlim,spacer=0,vline=[],freq=freq,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,20),textsize=16,hline=[])
    array_shape=x[2].shape
    fig=plt.figure(figsize=(10,6))                            #graphs of frequencies
    ax=fig.add_subplot(111)
    fig, axs= plt.subplots(nrows=8, ncols=1,figsize=(10,20),sharex=True,sharey=True)
    output_arr=np.array([])
    
    for i in range(1,array_shape[1]):
        x_axis=x[2][:,0]
        y_axis=x[2][:,i]*-1

        
        num_of_xsplits=round((x_axis.max()-x_axis.min())/seg_length) 
        splitx= np.array_split(x_axis, num_of_xsplits)  #actually splitting up the time
        splity= np.array_split(y_axis, num_of_xsplits) # splitting up the voltage as well

         
        
        blank_array= np.zeros([num_of_xsplits,2], dtype=float)  #creating an array of zeros on 2 columns
        for j,arr in enumerate(splitx):
            new_indexes = peakutils.indexes(splity[j], thres=thres, min_dist=min_dist) #using peak_detection library to find moments of peaks
            total_time = splitx[j].max()-splitx[j].min()    
            HR = 60*len(new_indexes)/total_time                #calculating HR by dividing total number of peaks by total time
            blank_array[j,0]=arr.mean()
            blank_array[j,1]=HR
            if save_fig_location!='':
                fig_temp=pyplot.figure(figsize=(10,6))
                print(i,j)
                print(new_indexes)
            
                if len(new_indexes)>0: 
                    pplot(splitx[j], splity[j], new_indexes)                               #peak graph
                    fig_temp.savefig(save_fig_location+'Channel'+str(i)+"_Segment_"+str(j)+".png")
                plt.close(fig_temp)
                
        if len(output_arr)==0: 
            output_arr=blank_array[:,0]
        output_arr=np.column_stack((output_arr,blank_array[:,1]))
              
        ax.plot(blank_array[:,0],blank_array[:,1],label='Channel ' + str(i))
        axs[i-1].plot(blank_array[:,0],blank_array[:,1])
        
        
    ax.legend(fontsize=10)
    return output_arr, axs, ax

