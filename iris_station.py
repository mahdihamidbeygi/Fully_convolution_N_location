from obspy import UTCDateTime, read, Stream
from ftplib import FTP
from glob import glob
from obspy.core.event import read_events
from obspy.clients.fdsn import Client
from mpl_toolkits.basemap import Basemap
from random import choices
import matplotlib.pyplot as plt
import os, sys, datetime
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core import AttribDict
from matplotlib.transforms import blended_transform_factory

additional_st = ['KEMF', 'NCHR', 'ENWF', 'CBC27', 'CQS64']
client = Client("iris")
tbeg=UTCDateTime("2010-01-01T00:00:00.00")
tend=UTCDateTime("2020-01-01T00:00:00.00")
filt = [0.001, 0.002, 25, 30]
lon0 = -135
lon1 = -118
lat0 = 46.5
lat1 = 55

def ADRM(eq,sta,path):
    ftp_address = 'ftp.seismo.nrcan.gc.ca'
    tb4ot = 60                    # downloading records from n seconds before origin time
    window = 180
    net = "CN"
    chan = ['HHZ','HHN','HHE']
    source = path
    if not os.path.exists(source+'/'):
        os.makedirs(source+'/')
    for event in [eq]:
        eventsignal = Stream()
        if os.path.exists('{}/CN.{}.mseed'.format(source,event)):
            continue
        tbeg = event - tb4ot
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
        delta = eventsignal[0].stats.delta
        eventsignal.trim(starttime = tbeg, endtime = tend - delta,pad = True,fill_value = 0)
        eventsignal.detrend('demean')
        eventsignal.detrend('linear')
        eventsignal.write('{}/CN.{}.mseed'.format(source,event),format='MSEED')
        


############# SELECTING STATIONS
stations = client.get_stations(network = '_REALTIME', starttime = tbeg, endtime = tend, startbefore = UTCDateTime("2010-01-01T00:00:00.00"), \
                               minlatitude = lat0, maxlatitude = lat1,\
                               minlongitude = lon0, maxlongitude = lon1, \
                                   level = 'channel')
for st in additional_st:
    stations += client.get_stations(network = 'NV', station = st, level = 'channel' ) 


stations = stations.select(network = 'CN', channel = '*', \
                           minlatitude = lat0, maxlatitude = lat1,\
                           minlongitude = lon0, maxlongitude = -122.5) + \
    stations.select(network = 'PB', channel = 'EH?', \
                           minlatitude = 46.7, maxlatitude = lat1,\
                           minlongitude = lon0, maxlongitude = lon1) + \
    stations.select(network = 'UW', channel = 'EN?',\
                           minlatitude = 46.5, maxlatitude = 49,\
                           minlongitude =-125, maxlongitude = -121) + \
        stations.select(network = 'CC') + stations.select(network = 'N[PV]', 
                                                          channel = 'HN?')
stations = stations.remove(network = 'US')

############ SIGNAL EXTRACTION AND WRITING TEXT FILES
# events = client.get_events(eventid = 610208500)
OT = UTCDateTime("2020-02-27T07:59:09")                     ## PUTTING ORIGIN TIME OF THE EARTHQUAKE THAT WE WANT TO HAVE SIGNALS FOR
t1 = OT - 5
t2 = t1 + 180
eq_lat = 48.052
eq_long = -123.107833
mag = 3.0
records = Stream()
for network in stations:
    for sta in network:
        st_lat = sta.latitude
        st_long = sta.longitude
        dist = gps2dist_azimuth(eq_lat, eq_long, st_lat, st_long)[0]
        for chan in sta:
            try: 
                print(network.code, sta.code, chan.code)
                                                #### GETTING SIGNALS CHANNEL BY CHANNEL 
                record = client.get_waveforms(network = network.code, station = sta.code, \
                                    channel = chan.code, location = '*', 
                                    starttime = t1, endtime = t2 ,attach_response = True)
                record.detrend('demean').detrend('linear').merge(method = 1, fill_value='interpolate')
                for tr in record:
                    tr.stats.distance = dist
                records += record.remove_response(pre_filt=filt,
                                                       water_level = 10,
                                                       taper = True,
                                                       taper_fraction = 0.00001)                    ## PREPROCESSING SIGNAL TO REMOVE INS. RESP. 
            except:
                # if network.code == 'CN':
                #     ADRM(OT, [st.code for st in network],'/home/mahdi/project/waveforms')
                #     continue
                print('No available Data for station {} with channel {}'.format(sta.code,chan.code))
                continue
        if False:
            with open("stationiris.dat", "w") as f:
                f.write("{} {} {} {} {} {}\n".format(sta.longitude, sta.latitude, network.code, sta.code, sta.start_date, sta.end_date))        # WRITING STATION INFO INTO A TEXT FILE
            f.close()
                                                     
records.write(filename = str(OT)+'.mseed', format='MSEED')                      ## SAVING SIGNALS INTO ONE MSEED FILE (WHOLE EARTHQUAKE SIGNALS)
############ PLOTTING STATIONS

if False:
    #stations.write('stations.dat', format="STATIONTXT")
    stations.plot(projection='local', resolution = 'f', method = 'basemap',
                  continent_fill_color = 'peru', water_fill_color = 'blue', 
                  legend = 'lower left',label = False,
                  color_per_network = True, outfile = 'station_info.tif' )          ## DRAWING A MAP SHOWING LOCATION OF STATIONS.
############ READING SIGNALS

# sts = []
# [[sts.append(st.code) for st in net] for net in stations]
# records = read('2020-02-27T07.59.09.000000Z.mseed')
for chan in ['??Z']:
    streams = records.select(channel = chan).sort(['distance'])

    # i = 0
    for i in range(0,len(streams),20):
        print(i)
        if len(streams[i:i+19]) == 0:
            continue
        else:
            signals = streams[i:i+19]
        fig = plt.figure()
        signals.plot(type='section', recordlength= t2 - t1, scale = 2.0,
                time_down=True, linewidth=.5, grid_linewidth=.25, show=False, 
                fig=fig)
        
        # Plot customization: Add station labels to offset axis
        ax = fig.axes[0]
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for tr in signals:
            ax.text(tr.stats.distance / 1e3, 1.0, tr.stats.station, rotation=270,
                    va="bottom", ha="center", transform=transform, zorder=10)
        plt.savefig('{}.{}.{}.tif'.format(OT,chan,i),
                          format = 'TIFF', dpi = 150)
        plt.show()

# signals.plot(type='section', orientation = 'horizontal', 
                  # reftime = t1, scale = 2.0,
                  # outfile = '{}.{}.tif'.format(OT,i),
                  # format = 'TIFF', dpi = 150)
# print(len(sts))
# UW = choices(sts,k=39)

