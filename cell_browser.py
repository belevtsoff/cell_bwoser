import cherrypy
import matplotlib.pyplot as plt
import StringIO
import urllib, base64
from matplotlib import mlab

import spike_sort
from spike_analysis import dashboard
from spike_sort.io.filters import BakerlabFilter
import sys,os
from jinja2 import FileSystemLoader, Environment
import tables

DATAPATH = os.environ['DATAPATH']

class Plotting:
    io_filter = BakerlabFilter(DATAPATH+'gollum_export.inf')
    sp_win = [-0.6, 0.8]
        
    def waveshapes(self, cell):
        ds = cell[:-5]
        sp = self.io_filter.read_sp(ds)
        spt = self.io_filter.read_spt(cell)
        sp_waves = spike_sort.extract.extract_spikes(sp, spt, self.sp_win)
        spike_sort.ui.plotting.figure()
        spike_sort.ui.plotting.plot_spikes(sp_waves)
        self.io_filter.close()
        
    def dashboard(self, cell):
        dashboard.show_cell(self.io_filter, cell)
    
    @staticmethod    
    def html_fig(fig=None):
        if not fig:
            fig = plt.gcf()
        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        plt.close('all')
        html_img =  "data:image/png;base64, " + urllib.quote(base64.b64encode(imgdata.getvalue()))
        return html_img
    
class HDF5filter:
    def __init__(self, h5fname):
        self.h5file = tables.openFile(h5fname, mode='a')
        
    def get_cached_string(self, cellid, name):
        path = cellid.split('/')
        path.append('cache')
        path = '/'.join(path)
        
        try:
            item = self.h5file.getNode(path, name)
            return item.read()
        except: return None
        
    def add_cached_string(self, cellid, name, str):
        path = cellid.split('/')
        path.append('cache')
        path = '/'.join(path)
        
        try: self.h5file.removeNode(path, name)
        except: pass
        
        self.h5file.createArray(path, name, str, createparents=True)
        self.h5file.flush()
        
    def add_spt(self, cellid, spt):
        pass
        

class HelloMpl:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.h5filter = HDF5filter('cell_db.h5')
        
        self.data = mlab.csv2rec(DATAPATH+'cell_db.csv')
        
        self.plotting = Plotting()
    
    def get_img_data(self, cellid, method, nocache=None):
        item = 'img_'+method
        img_data = self.h5filter.get_cached_string(cellid, item)
        
        if not img_data or nocache:
            getattr(self.plotting, method)(cellid)
            img_data = self.plotting.html_fig()
            self.h5filter.add_cached_string(cellid, item, img_data)
            
        return img_data
            
    
    @cherrypy.expose
    def index(self):
        return self.env.get_template('list.html').render(data=self.data)
    
    @cherrypy.expose
    def cell(self, cellid):
        visualize = ['dashboard','waveshapes']
        methods = list(set(visualize) & set(dir(self.plotting)))
        
        return self.env.get_template('cell.html').render(cellid=cellid, methods=methods)
    
    @cherrypy.expose
    def plot(self, method, cellid, nocache=None, clean=None, comment=None, reviewer=None):

        img_data = self.get_img_data(cellid, method, nocache)
        
        if reviewer: message = "You evaluation has been saved to the DB"
        else: message = ''
        
        return self.env.get_template('plot.html').render(cellid = cellid,
                                                         method = method,
                                                         img_data = img_data,
                                                         message = message)
        
cherrypy.quickstart(HelloMpl())
