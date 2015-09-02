'''
v.1.0

BaseMapTester3

| Copyright 2011 Crown copyright (c)
| Land Information New Zealand and the New Zealand Government.
| All rights reserved

This program is released under the terms of the new BSD license. See the LICENSE file for more information.

Created on 06/07/2014

@author: jramsay

Python script to query WMTS tile services on LDS recording tile fetch times over their entire zoom range. 

Usage:

python BaseMapTester5.py -u <simulated_users> [-w <width> -h <height>] [-r <reload_id>] [-v] [-h] {set|layer}<layer_id>
 
    Arguments
    ---------
    An identifier indicating the set|layer you want to test 
    OR one of the keywords ALL or RAND
            
    Options
    -------
    -u (--users) Number of users to simulate (thread count)
    -q (--sequential) Run users in Sequence or Parallel
    -p (--proxy) Use proxy with format host:port
    -h (--height) Number of tiles to fetch, vertical. Default=5
    -w (--width) Number of tiles to fetch, horizontal. Default=7
    -r (--reload) Reload/Replot a previously saved test
    -v (--version) Display version information
    -i (--info) Display this message
    -s (--show) Generate tile-success thumbnails. (Sets users to 1)

NB.1 API Keys (when enabled) should be saved in a file called ".key" in the same directory where this program is run.
this file should use the format key=ABCDEFGHIJKLIMNOPQRSTUVWXYZ12345678
'''

from urllib2 import HTTPError, base64, ProxyHandler
from datetime import datetime as DT, datetime
#from functools import wraps

import Image, ImageStat, ImageDraw
import urllib2
import StringIO
import random
import os
import sys
import re
import pickle
import getopt
import logging

import threading
import Queue
import numpy as NP

#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as PP
import matplotlib as mpl

class ImpossibleZoomLevelCoordinate(Exception): pass
class UnknownLandTypeRequest(Exception): pass
class UnknownConfigLayerRequest(Exception): pass

VER = 1.0
MAX_RETRY = 10
ZMAX = 20
ZMIN = 0
USERS = 16
LAND_THRESHOLD = 0.0005#0.0001 identifies ocean as parcel 0.001 cant find imagery in wlg_u/6 
WIDTH = 7
HEIGHT = 5
WHstr = str(WIDTH)+'x'+str(HEIGHT)
KEY = ''
B64A = ''
PARA = True
SHOW = False
NON_PC_LYR = 0

#Tile thumbnail size
TSIZE = 64

LOGFILE = 'BMT'
DEF_TILE_COLLECTION = 'RAND'#'basemap'
DEF_TILE_SET = 'RAND'#'colour'

#Default starting zoomlevel and start tiles
Z0I = ((0,20,0,0),)
Z3I = ((3,20,7,5),(3,20,7,4))

#in LY2 replace {id} with {id},{style} if we figure out what style means
STU = 'http://tiles-{cdn}.data-cdn.linz.govt.nz/services;key={k}/tiles/v4/set={id}{style}/{TileMatrixSet}/{TileMatrix}/{TileCol}/{TileRow}.png'
LY1 = 'http://tiles-{cdn}.data-cdn.linz.govt.nz/services;key={k}/tiles/v4/layer={id}{style}/{TileMatrixSet}{TileMatrix}/{TileCol}/{TileRow}.png'
LY2 = 'https://{cdn}.tiles.data.linz.govt.nz/services;key={k}/tiles/v4/layer={id}{style}/{TileMatrixSet}/{TileMatrix}/{TileCol}/{TileRow}.png'
GCU = 'https://data.linz.govt.nz/services;key={k}/wmts/1.0.0/set/{set}/WMTSCapabilities.xml'
#      type-refname = tc: url=tile URL
#                     tms=TileMatrixSet parameter
#                     sl=layer-refname=(layer-id, [style-id])
#                     ii=startcoordinates=z-start,z-end,x-start,ystart)
UU = {'imagery'  :{'url':STU,
                  'tms':'EPSG:3857',
                  'sl':{'National':(2,),},
                  'ii':Z3I},
      'basemap'  :{'url':STU,
                  'tms':'EPSG:3857',
                  'sl':{'Colour':(37,73),'Greyscale':(36,72)},
                  'ii':Z0I},
      'rural_nth':{'url':LY1,
                  'tms':'',
                  'sl':{'Northland_R':(1918,),},
                  'ii':((5,19,31,19),)},        
      'rural_akl':{'url':LY1,
                  'tms':'',
                  'sl':{'Auckland_R':(1769,),},
                  'ii':((4,19,15,9),)},
      'rural_ctl':{'url':LY1,
                  'tms':'',
                  'sl':{'Wellington_R':(1870,),'Manawatu_R':(1767,),'HawkesBay_R':(1778,),'BayOfPlenty_R':(1757,),'Taranaki_R':(1869,),'Waikato_R':(1872,)},
                  'ii':((3,19,7,5),)},
      'rural_est':{'url':LY1,
                  'tms':'',
                  'sl':{'Gisborne_R':(1722,),},
                  'ii':((4,19,15,9),)},
      'urban_akl':{'url':LY1,
                  'tms':'',
                  'sl':{'NorthShore_U':(1866,),},
                  'ii':((6,19,63,39),)},
      'urban_wlg':{'url':LY1,
                  'tms':'',
                  'sl':{'Wellington_U':(1871,),},
                  'ii':((6,19,63,40),)},
      'urban_chc':{'url':LY1,
                  'tms':'',
                  'sl':{'Christchurch_U':(1932,),},
                  'ii':((7,19,125,81),)},
      'urban_tmu':{'url':LY1,
                  'tms':'',
                  'sl':{'Timaru_U':(1927,),},
                  'ii':((7,19,124,81),)},
      'urban_bop':{'url':LY2,
                   'tms':'',
                   'sl':{'BayOfPlenty_U':(1753,),},
                   'ii':((5,19,31,19),)},
      'parcel'  :{'url':LY1,
                  'tms':'',
                  'sl':{'Parcel_81':(1571,81),'Parcel_82':(1571,82),'Parcel':(1571,),},
                  'ii':Z0I},      
      'pset'    :{'url':STU,
                  'tms':'',
                  'sl':{'wireframe':(69,),},
                  'ii':Z3I},
      'topo'    :{'url':LY1,
                  'tms':'',
                  'sl':{'t50':(767,),'t250':(798,)},
                  'ii':Z0I},
      'default' :{'url':LY1,
                  'tms':'',
                  'sl':{'layer':(0,),'set':(0,),},
                  'ii':Z0I},
      }


#-------------------------------------------------------------------------------------------
akf = '.key'
sty = 'auto'

FPATH = ''
TSTAMP = '{0:%y%m%d_%H%M%S}'.format(DT.now())
bmtlog = None#logging.getLogger(None)

#-------------------------------------------------------------------------------------------
#tmst = 'Contents/TileMatrixSet/ows:Title'
#tmsi = 'Contents/TileMatrixSet/ows:Identifier'
#tmi = 'Contents/TileMatrixSet/TileMatrix/ows:Identifier'

class TestRunner(object):
    def __init__(self):
        bmtlog.info('Starting BMT log')

    def testMultiUser(self,tc,ts): 
        #i dont think i need an input queue here, just used for join/wait
        bmtlog.info('Config parameters {}/{}'.format(tc,ts))
        
        if ts=='ALL' or tc=='ALL':
            #all configured layers
            rcs = TestRunner.getEveryTileSet() 
        elif ts=='RAND' or tc=='RAND':
            #randomly from all configured layers
            rcs = TestRunner.getRandomTileSet()
        else:
            rcs = TestRunner.parseUserArgsList(tc,ts)
            
        bmtlog.info('Client connections {}'.format(USERS))
        bmtlog.info('Layer request list {}'.format(rcs))
        

        ioq = {'in':Queue.Queue(),'out':Queue.Queue()}
        for ref,(rtc,rts) in enumerate(rcs):
            ioq['in'].put(ref)
            bmt = BaseMapTester(ioq['in'],ioq['out'])
                
            print 'TCTS',ref,rtc,rts, NON_PC_LYR if NON_PC_LYR else '' 
            zinit = random.choice(UU[rtc]['ii']) if len(UU[rtc]['ii'])>1 else UU[rtc]['ii'][0]
            bmt.setup(rtc,rts,*zinit)
            bmt.setDaemon(True)
            bmt.start()
            if not PARA:  
                ioq['in'].join()
                ioq['out'].join()
                
        if PARA:  
            ioq['in'].join()
            ioq['out'].join()
        
        print 'All queues joined'
        
        bmr = BaseMapResult(TSTAMP)
        bmp = BaseMapPickle(TSTAMP)
        bmp.setconf({'height':HEIGHT,'width':WIDTH,'npcl':NON_PC_LYR})
        while not ioq['out'].empty():
            qe = ioq['out'].get()
            bmp.append(qe)
            bmr.append(qe)
    
        bmp.dump()
        bmr.show()
        
    def loadTestResults(self,ref): 
        bmp = BaseMapPickle(ref)
        data = bmp.load()
        h,w,l = bmp.getconf()
        global HEIGHT
        HEIGHT = h
        global WIDTH
        WIDTH = w
        global NON_PC_LYR
        NON_PC_LYR = l
        bmr = BaseMapResult(ref,data)
        bmr.show()
        
    @classmethod
    def getEveryTileSet(cls):
        pairs = ()
        for user in range(USERS):
            select = [(u,UU[u]['sl'].keys()) for u in UU.keys()]
            UUa = sorted([(a,c) for a,b in select for c in b])
            pairs += (UUa[user%len(UUa)],)
        return pairs
    
    @classmethod
    def getRandomTileSet(cls):
        '''Randomly selects tileset weighting TS items equally'''
        pairs = ()
        for _ in range(USERS):
            weight = {u:len(UU[u]['sl']) for u in UU}
            expand = [(u,)*v for (u,v) in weight.items()]
            UUw = [val for sub in expand for val in sub]
            rtc = UUw[random.randint(0,len(UUw)-1)]
            rts = UU[rtc]['sl'].keys()[random.randint(0,len(UU[rtc]['sl'])-1)]
            pairs += ((rtc,rts),)
        return pairs
    
    @classmethod
    def parseUserArgsList(cls,tc,ts):
        '''Translates set|layer### to basemap|imagery... & Auckland_R|Timaru_U... format'''
        #get subsets for set/layer, find coll containing ts, find set containing ts
        if re.match('file', tc, re.IGNORECASE):
            lines = ()
            with open(ts) as handle:
                for entry in handle.readlines():
                    mat = re.match('(set|layer)(\d+)',entry)
                    if mat: lines += (TestRunner.parseUserArgs(mat.group(1),mat.group(2)),)
            global USERS
            USERS = len(lines)
            return lines
        elif re.match('ref', tc, re.IGNORECASE):
            return (TestRunner.validateUserArgs(tc,ts),)*USERS
        else:
            return (TestRunner.parseUserArgs(tc,ts),)*USERS

    @classmethod
    def parseUserArgs(cls,tc,ts):        
        '''Translates set|layer### to basemap|imagery... & Auckland_R|Timaru_U... format. NB Not so useful when style parameter is used'''
        #check whether user wants named configured collections or set/layer specifics
        
        #get subsets for set/layer, find coll containing ts, find set containing ts
        UUs = {u:v for u,v in UU.items() if tc in v['url']}#sets or layers
        x = [(u,v['sl']) for u,v in UUs.items() if int(ts) in [w[0] for w in v['sl'].values()]]
        #x = [(u,v) for u,v in x0 if int(ts) in [w[0] for w in v.values()]]
        #sort the tset results so the shortest result gets picked first, conventionally the one without style
        if x:
            y = sorted([i for i,j in x[0][1].items() if int(ts) in j])
        else:
            #if there is no x the layer/set wasnt found so return defaults
            global NON_PC_LYR
            NON_PC_LYR = ts
            return 'default', 'layer'
        return x[0][0],y[0]#x[1][0].y[0]
    
    @classmethod
    def validateUserArgs(self,r1,r2):
        '''If user provides configured name and layer make sure it exists in config array'''
        #r1=ref,r2=parcelSomething
        m1 = re.match('('+'|'.join(UU.keys())+')(\w+)',r2,re.IGNORECASE)
        if m1 and m1.group(2) in UU[m1.group(1)]['sl'].keys():
            return m1.group(1),m1.group(2)
        msg = 'Invalid Set/Layer Reference{}/{}'.format(r1,r2)
        bmtlog.critical(msg)
        raise UnknownConfigLayerRequest(msg)
    
class BaseMapPickle(object):
    
    def __init__(self,ref):
        self.ref = ref
        self.data = {}
        self.data['data'] = []
        
    def append(self,data):
        self.data['data'].append(data)
        
    def setconf(self,conf):
        '''Save parameters such as WxH'''
        for k in conf.keys():
            self.data[k] = conf[k]
        
    def getconf(self):
        return self.data['height'],self.data['width'],self.data['npcl']
        
    def dump(self):
        pdir = '{}{}'.format(FPATH,self.ref)
        if not os.path.exists(pdir): os.mkdir(pdir)
        pickle.dump(self.data,open('{}/{}.p'.format(pdir,self.ref),'wb'))
        
    def load(self):
        self.data = pickle.load(open('{}{}/{}.p'.format(FPATH,self.ref,self.ref),'rb'))
        return self.data['data']
        
class BaseMapTester(threading.Thread):
    
    def __init__(self,inq,outq):
        threading.Thread.__init__(self)
        self.inq = inq
        self.outq = outq
        self._stop = threading.Event()
        
    def setup(self,tc='basemaps',ts='colour',mn=0,mx=20,x1=0,y1=0):
        '''Parameter setup. 
        TC - Tile Collection, TS - Tile Set
        mn - Min zoom, mx - Max zoom
        x1 - Start X p-coord, y1 - Start Y p-coord'''
        self.tcol, self.tset, self.zmin, self.zmax, self.xinit, self.yinit = tc,ts,mn,mx,x1,y1
    
        
    def run(self):
        self.res = (self.tcol,self.tset)
        while not self.inq.empty():
            self.res += (self.testBaseMap(self.inq.get()),)
        self.close()
        
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
    
    def close(self):
        self.outq.put(self.res)
        print 'Queue {} loaded with {}/{}. '.format(self.outq.qsize(),self.res[0],self.res[1])
        bmtlog.info('Queue {} stopped for {}-{}'.format(self.outq.qsize(),self.res[0],self.res[1]))
        self.inq.task_done()
        self.outq.task_done()
    
    def testBaseMap(self,ref):
        dlist = ((ref,self.zmin,DT.now(),(0,0,0,0,0,0)),)
        tlist = {}
        xyz = (self.xinit,self.yinit,self.zmin)
        mr = MapRange(ref,self.tcol,self.tset)
        retry = 0
        zlev = self.zmin
        while zlev<self.zmax:
            nbrs = mr.getNeighbours(*xyz,width=WIDTH,height=HEIGHT)#get WxH tileset coords
            tlist[zlev] = mr.getAllTiles(nbrs)
            if SHOW: BaseMapResult._showTileArray(tlist[zlev])
    
            landkeys = [zk for zk in tlist[zlev] if tlist[zlev][zk]['mn'] > LAND_THRESHOLD and tlist[zlev][zk]['ex']]
            fail429 = len([zk for zk in tlist[zlev] if tlist[zlev][zk]['er']==429])
            fail500 = len([zk for zk in tlist[zlev] if tlist[zlev][zk]['er']==500])
            fail503 = len([zk for zk in tlist[zlev] if tlist[zlev][zk]['er']==503])
            failXXX = len([zk for zk in tlist[zlev] if tlist[zlev][zk]['er'] and tlist[zlev][zk]['er']<>500 and tlist[zlev][zk]['er']<>503])
            zero = len([zk for zk in tlist[zlev] if tlist[zlev][zk]['ex']])
            if landkeys:
                retry = 0
                print '{}# z={} c={},t={} - Success'.format(ref,zlev,xyz,len(landkeys))
                bmtlog.debug('{}# z={} c={},t={}'.format(ref,zlev,xyz,len(landkeys)))
                xyz = mr.selectTile(tlist[zlev], landkeys, 'INLAND')
                #print '{}# land tiles c={}'.format(ref,landkeys)
                #print '{}# selected land tile c={} m={} s={}'.format(ref,xyz,tlist[zlev][xyz]['mn'],tlist[zlev][xyz]['sd'])
                #tlist[zlev][xyz]['img'].show()
                xyz = mr.translate(*xyz)
            elif retry<MAX_RETRY:
                retry += 1
                print '{}# z={} c={},t={} - Shift and Retry {}'.format(ref,zlev,xyz,len(landkeys),retry)
                bmtlog.debug('{}# z={} c={},t={} - Shift and Retry {}'.format(ref,zlev,xyz,len(landkeys),retry))
                xyz = mr.shift(*xyz)
                zlev = xyz[2]
            else: 
                print '{}# z={} c={} t=0 - No Land Tiles'.format(ref,zlev,xyz)
                bmtlog.debug('{}# z={} c={} t=0 - Quit'.format(ref,zlev,xyz))
                bmtlog.error('Test Aborted - at Z={}'.format(zlev))
                return dlist
                #self.close()
            zlev += 1

            dlist += ((ref,zlev+1,DT.now(),(fail429,fail500,fail503,failXXX,zero,len(landkeys))),)
        bmtlog.info('{}# Test Complete - at Z={}'.format(ref,zlev))
        return dlist
    
class BaseMapResult(object):
    def __init__(self,ref,res=[]):
        self.ref = ref
        self.res = sorted(res)
        self.setup()

       
    
    def setup(self):
        mpl.rc('lines',linewidth=2)
        mpl.rc('font',size=10)
        mpl.rc('figure',figsize=(12,10),edgecolor='white')
        if self.res:
            self.reslen = max([len(s[2]) for s in self.res]) if self.res else 0
            self.cmap = PP.get_cmap('Spectral')
            self.colours = [self.cmap(float(i)/len(self.res)) for i in range(len(self.res))]
            
        self.fig = PP.figure()

        
    def offsetX(self,xx):
        return [x-1 for x in xx]
    
    def append(self,res):
        self.res.append(res)
        self.setup()
        
    def show(self):
        '''Select all the plots to generate'''
        self.plotRawUserTimeLine()
        self.plotRawUserTimeDiff()
        self.plotRawUserTimeAverageDiff()
        self.plotRawUserTimeMedianDiff()
        self.plotTimeDeltaHist()
        self.plotTileCounts()
        self.plot2DHistZoomTime()
        
    def _legend(self,j,seq,t=None):
        extra = '-'+NON_PC_LYR if NON_PC_LYR else ''
        if t: return 't{}-{}{}'.format(t[seq][0],seq,extra)
        return 'u{}-{}-{}{}'.format(j,seq[0],seq[1],extra)
    
    def plotRawUserTimeLine(self):
        fn = '{}{}/rawtime_{}.png'.format(FPATH,self.ref,self.ref)
        b = {}
        lgd = ()
        zero = DT(2000,1,1)
        for j,sequence in enumerate(self.res):
            minseq = min(sequence[2])
            lgd += (self._legend(j,sequence),)
            X = [i[1] for i in sequence[2]]
            #Y = [MDT.date2num(zero+(v[2]-minseq[2])) for v in sequence[2]]
            Y = [zero+(v[2]-minseq[2]) for v in sequence[2]]
            #xy = [(x,y) for x,y in zip(X,Y)]
            b[i] = PP.plot(self.offsetX(X),Y)#,color=self.colours[i])
        PP.legend(([b[bi][0] for bi in b]),lgd,loc=2)
        PP.title('Raw Zoom Timing / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('zoom level')
        PP.ylabel('time (h:m:s)')
        self._output(PP,fn)
        self.fig.clear()
        
    def plotRawUserTimeDiff(self): 
        #TODO. shift diff values left to x=0
        fn = '{}{}/rawdiff_{}.png'.format(FPATH,self.ref,self.ref)
        b = {}
        lgd = ()
        prev = [(x,y) for x,y in zip(range(self.reslen),self.reslen*[0,])][1:]
        for j,sequence in enumerate(self.res):
            #title
            lgd += (self._legend(j,sequence),)
            #extract and diff y values
            p1 = [v[2] for v in sequence[2]]
            p2 = p1[1:]
            #extract, calc and pair coordinates
            X = [i[1] for i in sequence[2]][1:]
            Y = [(p2[i]-p1[i]).seconds+(p2[i]-p1[i]).microseconds/1e6 for i in range(len(p1)-1)]
            xy = [(x,y) for x,y in zip(X,Y)]
            #set base value for seq
            B = [y for x,y in self.align(xy,prev)]
            #plot bar
            b[j] = PP.bar(self.offsetX(X),Y,bottom=B,color=self.colours[j])
            #store new stack value
            prev = self.stack(xy,prev)
        #-----------------------------
        PP.legend(([b[bi][0] for bi in b]),lgd,loc=2)
        PP.title('Raw Zoom Time Differences / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('zoom level')
        PP.ylabel('time (seconds)')
        self._output(PP,fn)
        self.fig.clear()
        
    def plotRawUserTimeAverageDiff(self): 
        fn = '{}{}/avgdiff_{}.png'.format(FPATH,self.ref,self.ref)
        b = {}
        t = dict()
        lgd = ()
        for seq1 in self.res:
            k = '{}-{}'.format(seq1[0],seq1[1])
            #extract and diff
            p1 = [v[2] for v in seq1[2]]
            p2 = p1[1:]
            #calc X and Y
            X = [i[1] for i in seq1[2]][1:]
            Y = [(p2[i]-p1[i]).seconds+(p2[i]-p1[i]).microseconds/1e6 for i in range(len(p1)-1)]
            #build xy pairs
            xy = [(x,y) for x,y in zip(X,Y)]
            #build dict of summed Y values and contributing line count with coll/set as key to aggregate with
            if k in t.keys():
                t[k] = (t[k][0]+1,self.stack(t[k][1],xy))
            else:
                t[k] = (1,xy)

        for j,seq2 in enumerate(t):
            #lgd += ('{}/{}'.format(seq2,t[seq2][0]),)
            lgd += (self._legend(j,seq2,t),)
            shift = 1.0/len(t)
            #calculate bar width and average Y values/clients
            X2 = [i[0]+(j*shift) for i in t[seq2][1]]
            Y2 = [y[1]/t[seq2][0] for y in t[seq2][1]]
            b[j] = PP.bar(self.offsetX(X2),Y2,width=shift,color=self.colours[j])
            #prev = [i+j for (i,j) in zip(delta,prev)]
        #-----------------------------
        PP.legend(([b[bi][0] for bi in b]),lgd,loc=2)
        PP.title('Raw Zoom Average Time Differences / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('zoom level')
        PP.ylabel('time (seconds)')
        self._output(PP,fn)
        self.fig.clear()
        
    def plotRawUserTimeMedianDiff(self): 
        fn = '{}{}/meddiff_{}.png'.format(FPATH,self.ref,self.ref)
        b = {}
        t = dict()
        lgd = ()
        for seq1 in self.res:
            k = '{}-{}'.format(seq1[0],seq1[1])
            #extract and diff
            p1 = [v[2] for v in seq1[2]]
            p2 = p1[1:]
            #calc X and Y
            X = [i[1] for i in seq1[2]][1:]
            Y = [(p2[i]-p1[i]).seconds+(p2[i]-p1[i]).microseconds/1e6 for i in range(len(p1)-1)]
            #build xy pairs
            xy = [(x,(y,)) for x,y in zip(X,Y)]
            #build dict with X and sorted associated Y values
            if k in t.keys():
                #append new value to result list
                t[k] = (t[k][0]+1,[(z1[0],(z1[1]+z2[1])) for z1,z2 in zip(t[k][1],xy)])
            else:
                t[k] = (1,xy)
                
        for kk in t.keys():
            #set median value in place of value list
            for n,col in enumerate(t[kk][1]):
                t[kk][1][n] = (col[0],self.median(col[1]))
            
        for j,seq2 in enumerate(t):
            #lgd += ('{}/{}'.format(seq2,t[seq2][0]),)
            lgd += (self._legend(j,seq2,t),)
            shift = 1.0/len(t)
            #calculate bar width and average Y values/clients
            X2 = [i[0]+(j*shift) for i in t[seq2][1]]
            Y2 = [y[1] for y in t[seq2][1]]
            b[j] = PP.bar(self.offsetX(X2),Y2,width=shift,color=self.colours[j])
            #prev = [i+j for (i,j) in zip(delta,prev)]
        #-----------------------------
        PP.legend(([b[bi][0] for bi in b]),lgd,loc=2)
        PP.title('Raw Zoom Median Time Differences / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('zoom level')
        PP.ylabel('time (seconds)')
        self._output(PP,fn)
        self.fig.clear()
        
    def plotTimeDeltaHist(self):
        fn = '{}{}/dlthist_{}.png'.format(FPATH,self.ref,self.ref)
        #lis = self.flatten(self.res)
        delta = [] 
        for sequence in self.res:
            p1 = [v[2] for v in sequence[2]]
            p2 = p1[1:]
            delta += [(p2[i]-p1[i]).seconds+(p2[i]-p1[i]).microseconds/1e6 for i in range(len(p1)-1)]
        #-----------------------------
        PP.hist(delta,50)
        PP.title('Tile Fetch-Time Histogram / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('seconds/layer')
        PP.ylabel('frequency')
        self._output(PP,fn)
        self.fig.clear()
        
    #fail,zero,land
    
    def plotTileCounts(self):
        defns = (('tilefail429','Tile Failure Count HTTP429'),
                 ('tilefail500','Tile Failure Count HTTP500'),
                 ('tilefail503','Tile Failure Count HTTP503'),
                 ('tilefail5XX','Tile Failure Count HTTP5XX'),
                 ('tileblank','Tile Blank Count'),
                 ('tileland','Tile Land Count'))
        
        for j,dd in enumerate(defns):
            self.plotCount(j,dd)
            
    def plotCount(self,pnum,deftxt):
        fn = '{}{}/{}_{}.png'.format(FPATH,self.ref,deftxt[0],self.ref)
        b = {}
        lgd = ()
        prev = [(x,y) for x,y in zip(range(self.reslen),self.reslen*[0,])][1:]
        for j,sequence in enumerate(self.res):
            #legend
            #lgd += ('{}-{}-{}'.format(j,sequence[0],sequence[1]),)
            lgd += (self._legend(j,sequence),)
            #extract, calc and pair coordinates
            X = [i[1] for i in sequence[2]]
            Y = [v[3][pnum] for v in sequence[2]]#500 errors
            xy = [(x,y) for x,y in zip(X,Y)]
            #set base value for seq
            B = [y for x,y in self.align(xy,prev)]
            #plot bar
            b[j] = PP.bar(self.offsetX(X),Y,bottom=B,color=self.colours[j])
            prev = self.stack(xy,prev)
        #-----------------------------
        PP.legend(([b[bi][0] for bi in b]),lgd,loc=2)
        PP.title('{} / Res({}), User({}{})'.format(deftxt[1],WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('zoom level')
        PP.ylabel('tile count')
        PP.xlim(0,max([mx for mx,my in prev]))
        self._output(PP,fn)   
        self.fig.clear()
        
 
    def plot2DHistZoomTime(self):
        fn = '{}{}/zthist_{}.png'.format(FPATH,self.ref,self.ref)
        delta = []
        zrnge = []
        for sequence in self.res:
            p1 = [v[2] for v in sequence[2]]
            p2 = p1[1:]
            delta += [float((p2[i]-p1[i]).seconds+(p2[i]-p1[i]).microseconds/1e6) for i in range(len(p1)-1)]
            zrnge += [v[0]*20 for v in sequence[2]][:-1]
        #-----------------------------
        #x = NP.random.randn(3000)-1
        x = NP.array(delta)
        #y = NP.random.randn(3000)*2+1
        y = NP.array(zrnge)
        h,xx,yy = NP.histogram2d(x,y,bins=20,range=[[0,200],[0,200]])
        extent = [xx[0], xx[-1], yy[0], yy[-1] ]
        
        PP.imshow(h.T,extent=extent,interpolation='bicubic',origin='lower')
        PP.colorbar()
        #-----------------------------
        #PP.legend(([b[bi][0] for bi in b]),t)
        PP.title('Zoom x Time 2D Histogram / Res({}), User({}{})'.format(WHstr,USERS,'p' if PARA else 's'))
        PP.xlabel('time (seconds)')
        PP.ylabel('zoom level')

        self._output(PP,fn)
        self.fig.clear()
        
    def stack(self,curr,prev):
        '''Add two datasets provided as pairs by matching x coords'''
        p = dict(prev)
        c = dict(curr)
        #stack([(1,100),(2,200)],[(2,2),(3,4)]) -> [(1,100),(2,202),(3,3)]
        return [(k,((p[k] if k in p else 0) + (c[k] if k in c else 0))) for k in set(p.keys()+c.keys())]
    
    def align(self,curr,prev):
        '''Aligns prev and curr data sets along X axis values discarding values not matching curr set'''
        p = dict(prev)
        c = dict(curr)
        #assumes prev has full key set
        return [(k,p[k] if k in p else 0) for k in set(c)]
    
    def median(self,lst):
        slst = sorted(lst)
        llen = len(lst)
        index = (llen - 1) // 2
    
        if (llen % 2):
            return slst[index]
        else:
            return (slst[index] + slst[index + 1])/2.0
        
    @classmethod
    def _output(cls,pobj,fn=None):
        pobj.savefig(fn, bbox_inches='tight') if fn else pobj.show()

    @classmethod        
    def _showTileArray(cls,tilearray):
        '''Static debugging method to show tile tracking progress'''
        z = tilearray.keys()[0][2]
        screensize = min(TSIZE*pow(2,z),TSIZE*WIDTH),min(TSIZE*pow(2,z),TSIZE*HEIGHT)
        screen = Image.new('RGB',screensize,'grey')
        blank = Image.new('RGB',2*(TSIZE,),'white')
        
        xx = sorted(set([a[0] for a in tilearray.keys()]))
        yy = sorted(set([a[1] for a in tilearray.keys()]))
        for x in xx:
            #line = ''
            for y in yy:
                print tilearray[(x,y,z)]['url']
                img = tilearray[(x,y,z)]['img']
                img.thumbnail(2*(TSIZE,))
                if tilearray[(x,y,z)]['mn']>LAND_THRESHOLD:
                    screen.paste(img,((x-min(xx))*TSIZE,(y-min(yy))*TSIZE))
                else:
                    screen.paste(blank,((x-min(xx))*TSIZE,(y-min(yy))*TSIZE))
                #if tilearray[(x,y,z)]['mn']>LAND_THRESHOLD:
                #    line += '+'
                #else: line += '-'
            #print line
        fn = '{}{}/grid_{}.png'.format(FPATH,TSTAMP,'.'.join([i[0]+str(i[1]) for i in zip('XYZ',tilearray.keys()[0])]))
        BaseMapResult._overlayTileGrid(screen)
        screen.save(fn)
        #screen.show()
        
    @classmethod
    def _overlayTileGrid(cls,img):
        xx,yy = img.size
        draw = ImageDraw.Draw(img)
        #verts
        for x in range(0,xx,TSIZE):
            draw.line((x,0, x,yy), fill=128)
        for y in range(0,yy,TSIZE):
            draw.line((0,y, xx,y), fill=128)
            
        
    #@classmethod
    #def flatten(cls,lis):
    #    return list(chain.from_iterable(item if isinstance(item, Iterable) and not isinstance(item, basestring) else [item] for item in lis))

class MapRange(object):
    
    def __init__(self,ref,tcol='default',tset='layer'):
        self.setTileCollection(tcol)
        self.setTileSet(tset)
        
        id = UU[tcol]['sl'][tset][0]
        self.ref = ref
        self.URL = UU[tcol]['url']
        self.TMS = UU[tcol]['tms']
        self.SORL = UU[tcol]['sl'][tset][0] if UU[tcol]['sl'][tset][0] else NON_PC_LYR
        self.STYLE = ',style={}'.format(UU[tcol]['sl'][tset][1]) if len(UU[tcol]['sl'][tset])>1 else ''
        #self.zlev = {i:{} for i in range(ZMIN,ZMAX)}

        
    def setTileSet(self,tset):
        '''Set the set/layer number'''
        self.tset = tset
        
    def setTileCollection(self,tcol):
        '''Set the type of tiles being probed, imagery, basemaps etc'''
        self.tcol = tcol
        
    def getBounds(self,t):
        '''get values for selection half width/height around centre'''
        return (t-1)/2,t/2+1
    
    def getNeighbours(self,x, y, z, width, height):
        '''Returns coordinates of all valid neighbours within WxH'''
        w,h = self.getBounds(width),self.getBounds(height)
        return [(a,b,z) for a in range(x-w[0],x+w[1]) for b in range(y-h[0],y+h[1]) if a>-1 and b>-1 and a<pow(2,z) and b<pow(2,z)]
    
    @classmethod
    def translate(cls,x,y,z,newz=None):
        '''Zooms in x and y coords zplus levels and increments z by zplus def 1'''
        zplus = newz-z if newz else 1
        return (x*pow(2,zplus),y*pow(2,zplus),z+zplus)
    
    @classmethod            
    def shift(self,x,y,z):
        '''Blindly(!) select a neighbouring tile if the zoomed centre tile doesnt return any valid land tiles'''
        return (abs(x+random.randint(-1,1)),abs(y+random.randint(-1,1)),z)
    
    
    @classmethod
    def randCDN(cls):
        '''Random a-d for tile server url i.e. tile-X'''
        return random.choice(map(chr, range(97, 101)))
    
    def getAllTiles(self,clist):
        imlist = {}
        tinq = Queue.Queue()
        totq = Queue.Queue()
        for cref,coords in enumerate(clist):
            params = {'coords':coords,'url':self.URL,'tms':self.TMS,'sorl':self.SORL,'st':self.STYLE}
            tinq.put(params)
            fetcher = TileFetcher(self.ref,cref,tinq,totq)
            fetcher.start()
        tinq.join()
        totq.join()
        
        while not totq.empty():
            imlist.update(totq.get())
        return imlist

    def selectTile(self,zlist,keys,criteria='RANDOM'):
        #high image mean and sd over varied terrain. high mean and low sd over bounding tiles
        kv = {k:(zlist[k]['mn']*zlist[k]['sd'])*random.random() for k in zlist.keys() if k in keys}
        #print '{} : {}'.format(criteria,kv)
        if criteria=='COAST':
            xyz = min(kv, key = kv.get)
        elif criteria=='INLAND':
            xyz = max(kv, key = kv.get)
        elif criteria=='LAND':
            pass
        elif criteria=='RANDOM':
            xyz = keys[random.randint(0,len(keys)-1)] 
        else:
            raise UnknownLandTypeRequest()
        return xyz
    
class TileFetcher(threading.Thread):
    def __init__(self,ref,cref,tinq,totq):
        threading.Thread.__init__(self)
        self.ref = ref
        self.cref = cref
        self.tinq = tinq
        self.totq = totq

    def run(self):
        self.totq.put(self.getTile())
        self.close()
        
    def close(self):
        self.tinq.task_done()
        self.totq.task_done()
        
    def getTile(self):
        '''Build the tile URL and fetch an image returning stats'''
        params = self.tinq.get()
        x,y,z = params['coords']
        url = params['url'].format(k=KEY,id=params['sorl'],style=params['st'],cdn=MapRange.randCDN(),TileMatrixSet=params['tms'],TileMatrix=z,TileCol=x,TileRow=y)
        req = urllib2.Request(url)
        #req.add_header("Authorization", "Basic {0}".format(B64A))
        retry = 1
        err = None
        while True:
            try:
                t1 = datetime.now()
                istr = urllib2.urlopen(req).read()
                t2 = datetime.now()
                img = Image.open(StringIO.StringIO(istr))
                t3 = datetime.now()
                istat = ImageStat.Stat(img)
                t4 = datetime.now()
                bmtlog.info('URL={},T-url={}, T-opn={}, T-stt={}'.format(url,t2-t1,t3-t2,t4-t3))
                isx = (istat.mean[0],istat.stddev[0],istat.extrema)
                break
            except HTTPError as he:
                print 'HTTP Error retrieving url {}\n{}'.format(url,he)
                emsg = re.search('HTTP Error (50\d)',str(he))
                if emsg and retry<MAX_RETRY:
                    print '{}.{}# Retrying {} - {}'.format(self.ref,self.cref,retry,(x,y,z))
                    err = int(emsg.group(1))
                    retry += 1
                else:
                    print 'Setting zero stats'
                    img = None
                    isx = ([0], [0], [(15,15)])
                    bmtlog.error('{}.{}# {} - {}'.format(self.ref,self.cref,he,url))
                    break
            except Exception as ue:
                print 'Unknown Error retrieving url {}\n{}\n'.format(url,ue)
                if retry<MAX_RETRY:
                    print '{}.{}# Retrying {} - {}'.format(self.ref,self.cref,retry,(x,y,z))
                    err = 0
                    retry += 1
                else:
                    print 'Setting zero stats'
                    img = None
                    isx = ([0], [0], [(15,15)])
                    bmtlog.error('{}.{}# {} - {}'.format(self.ref,self.cref,ue,url))
                    break
                
        return {(x,y,z):{'img':img,'mn':isx[0],'sd':isx[1],'ex':isx[2]<>[(15,15)],'er':err,'url':url}}
    
#------------------------------------------------------------------------------------------
def logger(lf,rlid=None,ll=logging.DEBUG,ff=2):
    formats = {1:'%(asctime)s - %(levelname)s - %(module)s %(lineno)d - %(message)s',
               2:'%(asctime)s - %(levelname)s - %(message)s',
               3:':: %(module)s %(lineno)d - %(message)s',
               4:'%(asctime)s,%(message)s'}
    
    log = logging.getLogger(lf)
    log.setLevel(ll)
    
    path = os.path.normpath(os.path.join(os.path.dirname(__file__),rlid if rlid else TSTAMP))
    if not os.path.exists(path):
        os.mkdir(path)
    df = os.path.join(path,'{}_{}.log'.format(lf.upper(),TSTAMP))
    
    fh = logging.FileHandler(df,'w')
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(formats[ff])
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    return log

def encode(auth):
    return base64.encodestring('{0}:{1}'.format(auth['user'], auth['pass'])).replace('\n', '')

def apikey(kfile):
    '''Read API Key from file'''
    return searchfile(kfile,'key')

def creds(cfile):
    '''Read CIFS credentials file'''
    return (searchfile(cfile,'username'),searchfile(cfile,'password'),searchfile(cfile,'domain','WGRP'))

def searchfile(sfile,skey,default=None):
    '''Generic config-format parameter reader'''
    value = default
    with open(sfile,'r') as h:
        for line in h.readlines():
            k = re.search('^{key}=(.*)$'.format(key=skey),line)
            if k: value=k.group(1)
    return value

def usage():
    print "Usage: python BaseMapTester_old.py -u <users> [-h <height> - w <width>] [-r <replay>] <layer_id|'RAND'|'ALL'>"
    print "ARGS\t{set|layer}layer_id. identifier 'layer' or 'set' followed by\n\tthe specific set/layer you want to test"
    print "\tRAND (keyword). 'u' Randomly selected set/layer from all\n\tconfigured sets/layers."
    print "\tALL (keyword). All configured sets/layers using 'u' threads." 
    print "OPTS\t-u <users>. Number of users to simulate (thread count)."
    print "\t-r <reload_id>. id string of dataset to reload/replot."
    print "\t-h <height>. Vertical tile count."
    print "\t-w <width>. Horizontal tile count."
    print "Version --version/-v."
    print "Help --info/-i"

def proxysetup(host,port):
    proxy = ProxyHandler({'http': '{0}:{1}'.format(host,port)})
    opener = urllib2.build_opener(proxy)
    #print 'Installing proxy',host,port
    urllib2.install_opener(opener)
    
def setup():
    '''Do any GLOBAL settings'''
    global KEY
    KEY = apikey(akf)
    
    #global B64A
    #B64A = encode({'user':u,'pass':p,'domain':d} if d else {'user':u,'pass':p})

def main():
    '''run test routines/simulations'''   
    reloadid = None
    tc,ts = DEF_TILE_COLLECTION,DEF_TILE_SET
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ivsw:h:r:u:p:", ["info","version","sequential","width=","height=","reload=","users=","proxy="])

    except getopt.error, msg:
        print msg
        #print "OPTS:",str(opts)
        #print "ARGS:",str(args)
        usage()
        sys.exit(2)
    
    for opt, val in opts:
        if opt in ("-i", "--info"):
            print __doc__
            sys.exit(0)
        elif opt in ("-v", "--version"):
            print VER
            sys.exit(0)
        elif opt in ("-u","--users"):
            global USERS
            USERS = int(val)
        elif opt in ("-q","--sequential"):
            global PARA
            PARA = False
        elif opt in ("-r","--reload"):
            reloadid = val 
        elif opt in ("-h","--height"):
            global HEIGHT
            HEIGHT = int(val) 
        elif opt in ("-w","--width"):
            global WIDTH
            WIDTH = int(val) 
        elif opt in ("-p","--proxy"):
            proxysetup(*val.split(':')) 
        elif opt in ("-s","--show"):
            print 'WARNING. Generating tile map, reverting to single user'
            global SHOW
            SHOW = True
            global USERS
            USERS = 1 
        else:
            print "unrecognised option:\n" \
            "-u (--users) Number of users to simulate (thread count)." \
            "-q (--sequential) Run users in Sequence or Parallel." \
            "-p (--proxy) Use proxy with format host:port."  \
            "-h (--height) Number of tiles to fetch, vertical. Default=5" \
            "-w (--width) Number of tiles to fetch, horizontal. Default=7" \
            "-r (--reload) Reload/Replot a previously saved test." \
            "-v (--version) Display version information" \
            "-i (--info) Display this message" \
            "-s (--show) Generate tile-success thumbnails. (Sets users to 1)"
            sys.exit(2)
                
    global bmtlog
    bmtlog = logger(LOGFILE,reloadid)
    
    tr = TestRunner()
    
    if reloadid:
        bmtlog.info('Reload dataset {}'.format(reloadid))
        tr.loadTestResults(reloadid)
        return
        
    global WHstr
    WHstr = str(WIDTH)+'x'+str(HEIGHT)
    
    if len(args)==0:
        usage()
        sys.exit(0)
        
    else:
        
        for arg in args:
            argmatch = re.match('(set|layer|file|ref)(\d+|\w+)', arg, re.IGNORECASE)
            if arg.lower() in ("rand", "random"):
                tc,ts = 'RAND','RAND'
            elif arg.lower() in ("all",):
                tc,ts = 'ALL','ALL'
            elif argmatch:
                tc,ts = argmatch.group(1).lower(),argmatch.group(2)
            else:
                print "Set/Layer definition required, use ALL|RAND|layer<layer_id>|set<set_id>|file<filename>|ref<UUcoll><UUset>"
                usage()
                sys.exit(0)
                
    #tc,ts can be eithe all or rand identifiers or a combo of set|layer+id
    bmtlog.info('Initial params TC:{} TS:{}'.format(tc,ts))
    tr.testMultiUser(tc,ts)
    return
    
    
if __name__ == '__main__':
    
    setup()
    main()
    print 'Finished'
