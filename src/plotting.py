import matplotlib.pyplot as plt
import StringIO
import urllib, base64

import spike_sort
from spike_analysis import dashboard, basic 
import numpy as np

class Visualize:
    def __init__(self, io_filter, root):
        self.io_filter = io_filter
        self.root = root
        
    def waveshapes(self, cell, sp_win=[-0.6, 0.8]):
        ds = cell[:-5]
        sp = self.io_filter.read_sp(ds)
        spt = self.io_filter.read_spt(cell)
        sp_waves = spike_sort.extract.extract_spikes(sp, spt, sp_win)
        spike_sort.ui.plotting.figure()
        spike_sort.ui.plotting.plot_spikes(sp_waves)
        self.io_filter.close()
        
    def dashboard(self, cell):
        dashboard.show_cell(self.io_filter, cell)


    def _get_patterns(self, cell):
        
        def _find_classes(trains,ev):
            """Classify spike trains"""
            tr_cl =[x[(x>ev[0]) & (x<ev[-1]) ] for x in trains]  
            cl = np.array([np.sum(2**(np.unique(np.searchsorted(ev,y))-1)) for y in tr_cl])
            cl[cl<0]=0
            return cl
        
        def _filter_spikes(spt,stim,win):
             i=np.searchsorted(stim,spt);
             spt2=(spt-stim[i-1]);
             spikes=spt[(spt2<win[1]) & (spt2>=win[0])]
             return spikes
        
        dataset = dashboard.read_dataset(self.io_filter, cell)
        spt = dataset['spt']
        stim = dataset['stim']
        ev = np.sort(self.root.h5filter.get_cached_string(cell,
                                                          "events"))
        dataset['events'] = ev

        win = [ev[0], ev[-1]]
        spt = _filter_spikes(spt, stim, win)
        trains = basic.SortSpikes(spt, stim, win) 
        cl = _find_classes(trains, ev)
        return dataset, cl

    @staticmethod
    def _dec2binstr(n, digits=None):
        '''convert denary integer n to binary string bStr'''
        bStr = ''
        if n < 0: raise ValueError, "must be a positive integer"
        if n == 0:
            bStr='0'
        while n > 0:
            bStr=bStr+ str(n % 2)
            n = n >> 1
        if digits:
            bStr=bStr.ljust(digits,'0')
        return bStr
    
    def spike_patterns(self, cell):
        
        dataset, cl = self._get_patterns(cell)
        spt = dataset['spt']
        stim = dataset['stim']
        
        plt.figure()
        plt.subplot(211)
        bins = np.arange(cl.min(), cl.max()+1)
        n, bins = np.histogram(cl, bins)
        n = n*1./np.sum(n)
        bins = bins[:-1]
        w = 0.8
        plt.bar(bins, n, width=w, fc='none')
        ndigits = np.ceil(np.log2(cl.max()))
        labels = map(lambda x: self._dec2binstr(x, ndigits), bins)
        plt.xticks(bins+w/2, labels)
        plt.xlabel('spike patterns')
        plt.ylabel('frequency')

        plt.subplot(212)
        for i in np.unique(cl):
            basic.plotPSTH(spt, stim[cl==i], rate=True, 
                           label=self._dec2binstr(i, ndigits))
        plt.legend()

    def spike_pattern_traces(self, cell):
        pass

        
def html_fig(fig=None):
    if not fig:
        fig = plt.gcf()
    imgdata = StringIO.StringIO()
    plt.savefig(imgdata, format='png')
    plt.close('all')
    html_img =  "data:image/png;base64, " + urllib.quote(base64.b64encode(imgdata.getvalue()))
    return html_img
