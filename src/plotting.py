import matplotlib.pyplot as plt
import StringIO
import urllib, base64

import spike_sort
from spike_analysis import dashboard

class Visualize:
    def __init__(self, io_filter):
        self.io_filter = io_filter
        
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

    def spike_patterns(self, cell):




        
def html_fig(fig=None):
    if not fig:
        fig = plt.gcf()
    imgdata = StringIO.StringIO()
    plt.savefig(imgdata, format='png')
    plt.close('all')
    html_img =  "data:image/png;base64, " + urllib.quote(base64.b64encode(imgdata.getvalue()))
    return html_img
