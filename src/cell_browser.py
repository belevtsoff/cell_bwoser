import cherrypy
from jinja2 import FileSystemLoader, Environment
from spike_sort.io.filters import BakerlabFilter

from filters import HDF5filter
from plotting import Visualize, html_fig
from matplotlib import mlab

import os

DATAPATH = os.environ['DATAPATH']

class HelloMpl:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.h5filter = HDF5filter('cell_db.h5')
        
        data_filter = BakerlabFilter(DATAPATH+'gollum_export.inf')
        self.vis = Visualize(data_filter)
        
        self.data = mlab.csv2rec(DATAPATH+'cell_db.csv')
    
    def get_img_data(self, cellid, method, nocache=None):
        item = 'img_'+method
        img_data = self.h5filter.get_cached_string(cellid, item)
        
        if not img_data or nocache:
            getattr(self.vis, method)(cellid)
            img_data = html_fig()
            self.h5filter.add_cached_string(cellid, item, img_data)
            
        return img_data
            
    
    @cherrypy.expose
    def index(self):
        return self.env.get_template('list.html').render(data=self.data)
    
    @cherrypy.expose
    def cell(self, cellid):
        visualize = ['dashboard','waveshapes']
        methods = list(set(visualize) & set(dir(self.vis)))
        
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
