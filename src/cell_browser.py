import cherrypy
from jinja2 import FileSystemLoader, Environment
from spike_sort.io.filters import BakerlabFilter

from filters import HDF5filter
from plotting import Visualize, html_fig
from matplotlib import mlab

import os

DATAPATH = os.environ['DATAPATH']

import json
import numpy as np

class HelloMpl:
    def __init__(self, env, data_filter):

        self.env = env
        
        self.h5filter = HDF5filter('cell_db.h5')
        
        self.data = mlab.csv2rec(DATAPATH+'cell_db.csv')
    
    def get_img_data(self, cellid, method, nocache=None):
        item = 'img_'+method
        img_data = self.h5filter.get_cached_string(cellid, item)
        
        if not img_data or nocache:
            getattr(self.visualize, method)(cellid)
            img_data = html_fig()
            self.h5filter.add_cached_string(cellid, item, img_data)
            
        return img_data
    
    @cherrypy.expose
    def cache_data(self, cellid, name, data):
        data = map(float, data.split(","))
        self.h5filter.add_cached_string(str(cellid), str(name), data)
        return 

    @cherrypy.expose
    def get_cached(self, cellid, name):
        data = self.h5filter.get_cached_string(cellid, name)
        if not data:
            json_data = {name:None}
        else:
            json_data = {name:data}
        return json.dumps(json_data)
    
    @cherrypy.expose
    def index(self):
        return self.env.get_template('list.html').render(data=self.data)
    
    @cherrypy.expose
    def cell(self, cellid):
        visualize = ['dashboard','waveshapes', 'spike_patterns']
        methods = list(set(visualize) & set(dir(self.visualize)))
        analyses = ["event_selector"] 
        return self.env.get_template('cell.html').render(cellid=cellid,
                                              methods=methods,
                                              analyse_methods=analyses)
    
    @cherrypy.expose
    def plot(self, method, cellid, nocache=None, clean=None, comment=None, reviewer=None):

        img_data = self.get_img_data(cellid, method, nocache)
        
        if reviewer: message = "You evaluation has been saved to the DB"
        else: message = ''
        
        return self.env.get_template('plot.html').render(cellid = cellid,
                                                         method = method,
                                                         img_data = img_data,
                                                         message = message)
        
from user_interface import UserInterface
data_filter = BakerlabFilter(DATAPATH+'gollum_export.inf')

conf = {'/js': {'tools.staticdir.on': True,
        'tools.staticdir.dir': '/Users/bartosz/SVN/personal/Analysis/cell_bwoser/src/js'}}

env = Environment(loader=FileSystemLoader('templates'))

root = HelloMpl(env, data_filter)
root.analyse = UserInterface(env, data_filter)
root.visualize = Visualize(data_filter, root)
cherrypy.quickstart(root, '/', config=conf)
