#!/usr/bin/env python3
from obspy import UTCDateTime, read, Stream
from ftplib import FTP
import numpy as np
import obspy as obs
import datetime
from glob import glob
import os
from obspy.clients.fdsn import client
import sys

ftp_address = 'ftp.seismo.nrcan.gc.ca'
tb4ot = 60                    # downloading records from n seconds before origin time
window = 180
net = "CN"
chan = ['HHZ','HHN','HHE']
filt = [0.001, 0.002, 25, 30]
sta = [sys.argv[1]]
eq = [sys.argv[2]]
print(eq,sta)
# source = "ADRM"
source = sys.argv[3]

if not os.path.exists(source+'/'):
    os.makedirs(source+'/')

# sta = []
# [ sta.append(str(line.split('|')[1])) for line in open("station.dat",'r')]
##slat = []
##[ slat.append(str(line.split('|')[2])) for line in open("station.dat",'r')]
##slon = []
##[ slon.append(str(line.split('|')[3])) for line in open("station.dat",'r')]
##selv = []
##[ selv.append(str(line.split('|')[4])) for line in open("station.dat",'r')]

# eq = []
# [ eq.append(line.split('|')[1]) for line in open('catalog.dat','r')]

i = 1
for event in eq:
    eventsignal = Stream()
    if os.path.exists('{}/CN.{}.{}.mseed'.format(source,sta[0],event)):
        i += 1
        continue
    tbeg = UTCDateTime(event) - tb4ot
    tend = tbeg + window
##    try :
    print(event, tbeg.strftime('%Y/%m/%d'))
    daydir = 'wfdata/CN/'+tbeg.strftime('%Y/%m/%d')+'/'

    ftp = FTP(ftp_address)
    ftp.login()
    ftp.cwd(daydir)

    lfiles = []
    ftp.retrlines('LIST',lfiles.append)
    files = [f[55:] for f in lfiles]
    mfiles=[s for s in files if any(st in s for st in sta) 
    and any(chn in s for chn in chan)]
    print(lfiles)
    print(mfiles)
    for record in mfiles: 
        print(record)
        if os.path.exists(record):
            continue
        with open(record, 'wb') as fpp: 
       	    res=ftp.retrbinary('RETR '+record,fpp.write)
    # -- Close ftp connection
    ftp.quit()
    
    [eventsignal.append(read(file).merge(fill_value=0)[0]) for file in mfiles]
    print(eventsignal)
    delta = eventsignal.stats.delta
    eventsignal.trim(starttime = tbeg, endtime = tend - delta,pad = True,fill_value = 0)
    eventsignal.detrend('demean')
    eventsignal.detrend('linear')
##    eventsignal.remove_response(pre_filt = filt, water_level = 10, \
##                                taper = True, taper_fraction = 0.00001)
    eventsignal.write('{}/CN.{}.{}.mseed'.format(source,sta,eq),format='MSEED')
    i += 1
##    except :
##        continue
