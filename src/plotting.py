import matplotlib.pyplot as plt
import StringIO
import urllib, base64

import spike_sort
from spike_analysis import dashboard, basic 
import numpy as np

from spike_sort.core.extract import ZeroPhaseFilter
from matplotlib.transforms import blended_transform_factory

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


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
        try:
            win = [ev[0], ev[-1]]
        except IndexError:
            ev = [0, 30]
            win = ev
        
        dataset['events'] = ev
        
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
        """Show distribution and PSTH of spike patterns.
        
        Requires selection of spike windows in the Event Selector.
        """
        
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

    def pattern_traces(self, cell, contact=1, n_traces=100, subtract_mean='True'):
        """Show raw traces from microelectrodes for different spike
        patterns.
        
        Requires selection of spike windows in the Event Selector.

        **Extra parameters**:

        * `contact` (int) -- index of tetrode contact to use (default
          0)
        * `n_traces` (int) -- number of traces  to plot (default 100)
        * `subtract_mean` (True of False) -- subtract mean from traces
        (default True)
        """
        
        contact = int(contact)
        subtract_mean = str2bool(subtract_mean)
        
        dataset, cl = self._get_patterns(cell)
        spt = dataset['spt']
        stim = dataset['stim']
        ev = dataset['events']
        if ev is None or len(ev)<2:
            ev = [0, 30]
        win = [ev[0], ev[-1]]
        sp = self.io_filter.read_sp(cell)
        #filter = ZeroPhaseFilter('cheby2', f_band)
        #sp = spike_sort.extract.filter_proxy(sp, filter)

        fig=plt.figure()
        n_patterns = len(np.unique(cl))
        ndigits = np.ceil(np.log2(cl.max()))
        ax=None
        stim_dict = {'data':stim}
        sp_traces = spike_sort.extract.extract_spikes(sp,
                                                      stim_dict,
                                                      win,
                                                      contacts=contact)
        if subtract_mean:
            sp_traces['data']=sp_traces['data']-sp_traces['data'].mean(1)[:,None,:]
        
        for i,c in enumerate(np.unique(cl)):
            ax=plt.subplot(n_patterns, 1, i+1, frameon=False,
                           sharex=ax)
            traces = sp_traces['data'][:,cl==c,0]
            plt.plot(sp_traces['time'], traces[:,:n_traces], 'k-')
            plt.text(0.9, 0.8, self._dec2binstr(c, ndigits),
                     transform=ax.transAxes)
            plt.yticks([])
            plt.setp(ax.xaxis.get_ticklabels(),visible=False)
            print sp_traces['data'].shape
        plt.xticks(ev)
        plt.setp(ax.xaxis.get_ticklabels(),visible=True)
        trans = blended_transform_factory(ax.transData,
                                          fig.transFigure)
        plt.vlines(ev, 0, 1, color='r', transform=trans, clip_on=False)

        pass

        
def html_fig(fig=None):
    if not fig:
        fig = plt.gcf()
    imgdata = StringIO.StringIO()
    plt.savefig(imgdata, format='png')
    plt.close('all')
    html_img =  "data:image/png;base64, " + urllib.quote(base64.b64encode(imgdata.getvalue()))
    return html_img
