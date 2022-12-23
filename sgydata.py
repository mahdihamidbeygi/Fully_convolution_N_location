import struct
import numpy as np
import random
import math
from obspy import read
from numpy.random import normal

def norm_sgy(data):                 ## removing mean value 
    maxval=max(max(data))
    minval=min(min(data))
    print(maxval,minval)
    return [[(float(i)-minval)/(float(maxval-minval)+0.01e-100) for i in j] for j in data]
def norm_sgy1(data):                ## normalizing to max value in three components
    maxval=max([max([abs(i) for i in j]) for j in data])
    #minval=min(min(data))
    #return [[(float(i)-min(j))/float(max(j)-min(j)) for i in j] for j in data]
    #return [[float(i)/(float(max(j))+0.01e-100) for i in j] for j in data]
    return [[float(i)/(maxval+0.01e-100) for i in j] for j in data]
def read_sgy(sgynam='test.sgy'):            ## reading sgy file which is binary file
    data1 = []
    tr = np.load(sgynam, allow_pickle=True)
    nr = [210]
    n = nr[0] - len(tr)
    nsmp = [4096]
    [data1.append(record[:nsmp[0]]) for record in tr]
    [data1.append(0*tr[i][:nsmp[0]]) for i in range(n)]
    data = []
    [data.append(np.append(signal,0*tr[0][:(nsmp[0] - len(signal))])) if len(signal) < nsmp[0] else data.append(signal) for signal in data1]
    data = np.nan_to_num(data)
    return nr,nsmp,data;
    
def loca_img_xyz(xr=[0.25,0.01,24],yr=[-0.2,0.013,32],zr=[3.07,0.01,18],xyz=[0.4,0.0,3.12],r=0.0005,rtz=(11.12/2.0)**2):       ## computing gaussian distribution 
    img=[]
    for i in range(0,xr[2]):
        x = xr[0]+xr[1]*i
        tmp1=[]
        for j in range(0,yr[2]):
          y=yr[0]+yr[1]*j
          tmp2=[]
          for k in range(0,zr[2]):
              z=zr[0]+zr[1]*k
              ftmp=(x-xyz[0])*(x-xyz[0])+(y-xyz[1])*(y-xyz[1])+rtz*(z-xyz[2])*(z-xyz[2])/(111.19)**2
              tmp2=tmp2+[math.exp(-0.5*ftmp/r)]
          tmp1.append(tmp2)
        img.append(tmp1);
    return img;

def shuffle_data(data,ydata,seed,shuffle):                  ## shuffling input and its label into two row vectors. $$$ do not train in a sort of stations and records
    if shuffle == 'false':
        return data,ydata;
    index=[i for i in range(len(ydata))]
    random.seed(seed)
    random.shuffle(index)
    data = [data[i] for i in index]                         
    ydata = [ydata[i] for i in index]                       
    return data,ydata
    
def load_sgylist_xyz1(sgylist=['./path/','./ok_syn1/sgylist.txt'],sgyr=[0,-1,1],xr=[3913.880-25.0,14.000,20],yr=[-10896.620-25,15.000,20],zr=[100001.000-3,3.00,10],r=500.000,rtz=(11.12/2.0)**2,
                      shuffle='true',shiftdata=[list(range(-5,2)),1]):          ## reading sgy files based on the text file provided in the main path (training sample.txt)
    #nx,ny,stn=read_stn(stnnam)
    with open(sgylist[1],'r') as f:
        lines=f.readlines()
    lines=lines[sgyr[0]:sgyr[1]:sgyr[2]]+[lines[sgyr[1]]];
    data= []                    ## input to NN
    ydata=[]                    ## labels for input 
    #eventnam=[]
    for i in range(0,len(lines)):
       line1=lines[i].split()
       sgynam=sgylist[0]+line1[0].strip()
       loca=[float(num) for num in line1[1:4]]
       nr,nsmp,data1 = read_sgy(sgynam);                            ## number of records (nr), number of samples (nsmp) and data existing in the sgy file. 
       img=loca_img_xyz(xyz=[loca[0],loca[1],loca[2]],xr=xr,yr=yr,zr=zr,r=r,rtz=rtz)
       print(i,sgynam)
#       data1=np.clip(np.nan_to_num(np.array(data1)),-1.0e-2,1.0e-2).tolist()                ## clearing data ( nan, inf)
       
       
       # for i in range(nr[0]):
       #     for j in range(nsmp[0]):
       #         if data1[i][j] <= -1.0e-2:
       #             data1[i][j] = -1.0e-2
       #         elif data1[i][j] >= 1.0e-2:
       #             data1[i][j] = 1.0e-2
       #         else:
       #             continue
       # data1 = list(data1)
    #       data1 = prep_data(data1,stn,nx,ny)
       if nr != 0:
         data1=norm_sgy1(data1)         ## normalizing to max value
         data1=[[[data1[ir][j],data1[ir+1][j],data1[ir+2][j]] for j in range(nsmp[0])] for ir in range(0,nr[0],3)]      ## reshape data placing 3 comp in a row vector
         data.append(data1)                     ## input to the NN
         ydata.append(img);                     ## Labels for the input in NN
         #eventnam.append(sgynam)
       else:
         print('1 event sgy not found')
    if shiftdata[1]>0:                              
        data1,ydata1=augment_data2(data=data,ydata=ydata,shiftdata=shiftdata);          ## shifting data 
        data=data+data1;
        ydata=ydata+ydata1;

    data,ydata=shuffle_data(data,ydata,1,shuffle);                      ## shuffling input and its label,(data, ydata) because of not training NN based on the a specific order of stations
    data=np.array(data)
    ydata=np.array(ydata, dtype=np.float32)
    return data,ydata
   
def augment_data2(data=[],ydata=[],shiftdata=[list(range(-5,2)),1]):
   #     data_out,ydata_out = cut_trace(icut=par[0],data=data,ydata=ydata);
        data1=[];
        ydata1=[];
        nsmp=len(data[0][0])
        for i in range(len(ydata)):
            random.seed(i);
            its=random.sample(shiftdata[0],shiftdata[1]);           ## choosing k between N number
            for j in range(0,len(its)):
                if its[j]<0:
                    data_tmp=[ftmp[nsmp+its[j]:]+ftmp[0:nsmp+its[j]] for ftmp in data[i]]           # shifting data
                else:
                    data_tmp=[ftmp[its[j]:]+ftmp[0:its[j]] for ftmp in data[i]]                 # Shifting data
                ydata_tmp=ydata[i];
                data1=data1+[data_tmp];
                ydata1=ydata1+[ydata_tmp];
        return data1,ydata1;

if __name__ == '__main__':
#     nr,nsmp,data=read_sgy()
#     print nr,nsmp,data[1:nsmp[0]]
     # numsgy,data,ydata=load_sgylist()
     # print numsgy,len1,len(data),data[1],data.shape,ydata.shape
     # img=loca_img()
     # print ydata;
     # print('mask', ydata[0][10])
     import scipy.io as sio
     # print(read_stn('ok_syn1/ok.stn'))
     # nam,data,ydata=load_sgylist_xyz1()
     nams,data,ydata=load_sgylist_xyz1(sgylist='loca_ok2_sgylist_largercv_train_n1013.txt',shuffle='false',
            sgyr=[0,5,1],xr=[39.0,0.1,160],yr=[-135,0.1,150],zr=[-5,1,80],r=5,twin=list(range(100,800)),asize=0,
            shiftdata=[list(range(20,50))+list(range(-200,-20)),0],doubleevent=[list(range(-200,-150)),1])
     #print(data,data.shape)[list(range(20,50))+list(range(-200,-20)),1]
     # sio.savemat('D:/cnnloca/test.mat',{'data':data})


