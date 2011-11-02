import matplotlib.pyplot as plt
import StringIO
import urllib, base64

import spike_sort
from spike_analysis import dashboard, basic 
import numpy as np

from matplotlib.transforms import blended_transform_factory

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


class Visualize:
    def __init__(self, io_filter, root):
        self.io_filter = io_filter
        self.root = root
        
    def waveshapes(self, cell, win='-0.6,0.8', n_traces=300):
        """Plot spike waveshapes on all contacts.

        **Extra parameters:**

        * `win` (float,float) -  plotting time window (default -0.6,0.8)
        * `n_traces` (int) - number of traces to plot (default 300)
        """

        sp_win = map(float, win.split(','))
        n_traces = int(n_traces)
        ds = cell[:-5]
        sp = self.io_filter.read_sp(ds)
        spt = self.io_filter.read_spt(cell)

        n_spikes = len(spt['data'])
        
        i = np.random.randint(0, n_spikes, n_traces)
        spt['data'] = spt['data'][i]

        sp_waves = spike_sort.extract.extract_spikes(sp, spt, sp_win)
        spike_sort.ui.plotting.figure()
        spike_sort.ui.plotting.plot_spikes(sp_waves)
        self.io_filter.close()
        
    def timeline(self, cell, win='-0.6,0.8', n_traces=100, n_splits=4):
        """Change of spike waveforms in time (from left to right).
        Last column shows means spike wavefroms in all other columns
        (colour coded)

        **Extra parameters: **
        
        * `win` (float,float) -  plotting time window (default -0.6,0.8)
        * `n_traces` (int) - number of traces to plot (default 300)
        * `n_splits` (int) - number of time frames the dataset is divided
                             into (default 4)
        """

        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

        sp_win = map(float, win.split(','))
        n_traces = int(n_traces)
        n_splits = int(n_splits)
        sp = self.io_filter.read_sp(cell)
        spt = self.io_filter.read_spt(cell)
        
        n_spikes = len(spt['data'])
        inc = n_spikes/n_splits
        n_chans, _ = sp['data'].shape
         
        fig=plt.figure()
        axs = [None for i in range(n_chans)]
        for split in xrange(n_splits):
            i = np.random.randint(inc*split, inc*(split+1), n_traces)
            spt_split = {'data': spt['data'][i]}
            sp_waves = spike_sort.extract.extract_spikes(sp, spt_split, sp_win)
            print sp_waves['data'].shape
            for chan in xrange(n_chans):
                ax = plt.subplot(n_chans, n_splits+1, chan*(n_splits+1)+split+1,
                            frameon=False, sharey=axs[chan])
                plt.plot(sp_waves['time'], sp_waves['data'][:,:,chan],
                         colors[split], alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                axs[chan] = ax
                # subplots with means
                plt.subplot(n_chans, n_splits+1,
                            (chan+1)*(n_splits+1))
                plt.plot(sp_waves['time'],
                         sp_waves['data'][:,:,chan].mean(1),
                         colors[split])

        for chan in xrange(n_chans):
            ax = plt.subplot(n_chans, n_splits+1, (chan+1)*(n_splits+1))
            ax.set_frame_on(False)
            #ax.set_ylim(axs[chan].get_ylim())
            ax.set_xticks([])
            ax.set_yticks([])

        plt.text(0.5, 0.05, 'time', ha='center', transform=fig.transFigure)
        plt.text(0.05, 0.5, 'channels', va='center',rotation=90, transform=fig.transFigure)

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
        dataset['spt'] = spt
        
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
        bins = np.arange(cl.min(), cl.max()+2)
        n, _ = np.histogram(cl, bins)
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
        
        # remove 'out of range' stimuli
        T_max = (sp['data'].shape[1])*1./sp['FS']*1000.
        b_idx = ((stim+win[0])>0) & ((stim+win[1])<T_max)
        stim = stim[b_idx]
        cl = cl[b_idx]
        
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
            plt.plot(sp_traces['time'], traces[:,:n_traces], 'k-',
                     alpha=0.2)
            plt.text(0.9, 0.8, self._dec2binstr(c, ndigits),
                     transform=ax.transAxes)
            plt.yticks([])
            plt.setp(ax.xaxis.get_ticklabels(),visible=False)
        
        plt.xticks(ev)
        plt.setp(ax.xaxis.get_ticklabels(),visible=True)
        trans = blended_transform_factory(ax.transData,
                                          fig.transFigure)
        plt.vlines(ev, 0, 1, color='r', transform=trans, clip_on=False)
    
    def pattern_spike_waveforms(self, cell, contact=0, n_spikes=100,
                                mean='False'):
        """Waveforms of spikes emitted in different spike patterns
        (shown in separate subplots) and spike windows (shown in
        different colors).

        Requires selection of spike windows in the Event Selector.

        **Extra parameters**:

        * `contact` (int) -- index of tetrode contact to use (default
        0),
        * `n_spikes` (int) -- number of spikes to plot (default 100)
        * `mean` (bool) -- plot only means (default False)
        """

        def which_window(spt, stim, ev):
            bWin = np.vstack([spike_in_win(spt, stim, [ev[i], ev[i+1]]) 
                           for i in range(len(ev)-1)])
            cl = bWin.argmax(0) if len(ev)>2 else bWin[0,:]*1
            return cl

        def spike_in_win(spt, stim, win):
            i = np.searchsorted(stim, spt)-1
            sp_pst = spt-stim[i]
            bool = ((sp_pst>win[0]) &( sp_pst<win[1]))
            return bool
        
        contact = int(contact)
        mean = str2bool(mean)
        win = [-1, 2]
        colors = ['r', 'b', 'g', 'y']

        dataset, cl = self._get_patterns(cell)
        spt = dataset['spt']
        stim = dataset['stim']
        ev = dataset['events']
        sp = self.io_filter.read_sp(cell)
        
        sp_dict = spike_sort.extract.extract_spikes(sp,
                                                      {"data": spt},
                                                      win,
                                                      contacts=contact)
        sp_time = sp_dict['time']
        sp_waves = sp_dict['data'][:,:, 0]
        #spt = spt[:sp_waves.shape[1]]
        stim_idx = stim.searchsorted(spt)-1
        sp_cl = cl[stim_idx]
        sp_window_lab = which_window(spt, stim, ev)
        labels = np.unique(cl)
        axes_list = []
        i_max = np.abs(sp_time).argmin()
        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15)
        ax=None
        min, max = sp_waves[i_max,:].min(),sp_waves[i_max,:].max()
        x = np.linspace(min, max, 100)
        for i,l in enumerate(labels):
            ax=plt.subplot(1, len(labels), i+1, frameon=False, sharey=ax)
            sp_win_waves = sp_waves[:, sp_cl==l]
            if sp_win_waves.shape[1]>0:
                idx = np.random.rand(sp_win_waves.shape[1]).argsort()
                waves_sh = sp_win_waves[:,idx[:n_spikes]]
                window_lab = sp_window_lab[:, sp_cl==l][idx[:n_spikes]]
                for w_l in np.unique(window_lab):
                    waves = waves_sh[:, window_lab==w_l]
                    if mean:
                        waves = waves.mean(1)
                        lw = 1.
                    else:
                        lw = 0.1
                    plt.plot(sp_time, waves,colors[w_l],lw=lw)
                
            plt.xlabel(self._dec2binstr(l,3))
            plt.xticks([])
        ax_annotate = fig.add_subplot(2,1,1, sharey=ax, frameon=False)
        plt.xticks([])
        plt.yticks([])
        ylims = plt.ylim()
        plt.axhline(0, lw=0.5, color='k')
        
def html_fig(fig=None):
    if not fig:
        fig = plt.gcf()
    imgdata = StringIO.StringIO()
    plt.savefig(imgdata, format='png')
    plt.close('all')
    html_img =  "data:image/png;base64, " + urllib.quote(base64.b64encode(imgdata.getvalue()))
    return html_img
