from obspy import UTCDateTime, read, Stream
from obspy.clients.fdsn import Client
import os
from random import choices


tb4ot = 5                    # downloading records from n seconds before origin time
window = 102.4
net = "CN"
chan = "HH"
filt = [0.001, 0.002, 25, 30]

source = "IRIS"
sta = []
[ sta.append(str(line.split('|')[1])) for line in open("station.dat",'r')]
eq = []
[ eq.append(line.split('|')[1]) for line in open('catalog.dat','r')]

if not os.path.exists(source+'/'):
    os.makedirs(source+'/')

client = Client(source)
i = 1
for event in eq :
    if os.path.exists(source + '/event{}.mseed'.format(i)):
        i += 1
        continue    
    eventsignal = Stream()
    tbeg = UTCDateTime(event) - tb4ot
    tend = tbeg + window
    for st in sta :
        try:
            print(event,st, tbeg, tend)
            tr = client.get_waveforms(network = net, station = st, channel = chan + '?',\
                                      starttime = tbeg, endtime = tend, \
                                      location = False, attach_response = True)
        except :
            tr = Stream()
            sta2 = []
            [ sta2.append(str(line.split('|')[1])) for line in open("station2.dat",'r')]
            while len(tr) == 0:
                st = choices(sta2,k=1)[0]
                sta2.remove(st)
                if st in [sts.stats.station for sts in eventsignal]:
                    continue
                try:
                    tr = client.get_waveforms(network = net, station = st, channel = chan + '?',\
                                  starttime = tbeg, endtime = tend, \
                                  location = False, attach_response = True)
                except:
                    continue
            print(' ####### station {} is chosen over the previous one'.format(st))
        tr.merge(method = 1, fill_value = 'interpolate')
        tr.detrend('demean')
        tr.detrend('linear')
        tr.remove_response(pre_filt = filt, water_level = 10,\
                               taper = True, taper_fraction = 0.00001)
        eventsignal += tr
    eventsignal.write(source + '/event{:03d}.mseed'.format(i),format = 'MSEED')
    i += 1
    
