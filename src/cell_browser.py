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

from utils import doc2html

def append_column(data, type):
    dtype = data.dtype.descr
    dtype.append(('score', np.int8))
    new_data = np.empty(data.shape, dtype=dtype)

    for name, type in data.dtype.descr:
        new_data[name] = data[name]
    return new_data


class HelloMpl:
    def __init__(self, env, data_filter):

        self.env = env
        
        self.h5filter = HDF5filter('cell_db.h5')
        
        self.data = mlab.csv2rec(DATAPATH+'cell_db.csv')
        self.data = append_column(self.data, ('score', np.int8))
   
    def update_data(self):

        for i in range(len(self.data)):
            cell_id = self.data['id'][i]
            score = self.h5filter.get_cached_string(cell_id, 'score')
            if score:
                self.data['score'][i] = int(score[0])
            else:
                self.data['score'][i] = 0
            comment = self.h5filter.get_cached_string(cell_id, 'comment') 
            if comment:
                self.data['comment'][i] = comment
            else:
                comment = str(self.data['comment'][i])
                self.h5filter.add_cached_string(cell_id, 'comment', comment)         

    def get_img_data(self, cellid, method, nocache=None, **opts):
        item = 'img_'+method
        img_data = self.h5filter.get_cached_string(cellid, item)
        
        if not img_data or nocache or opts:
            getattr(self.visualize, method)(cellid, **opts)
            img_data = html_fig()
            self.h5filter.add_cached_string(cellid, item, img_data)
            
        return img_data
    
    @cherrypy.expose
    def cache_data(self, cellid, name, data):
        try:
            data = map(float, data.split(","))
        except ValueError:
            data = str(data)
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
        self.update_data()
        return self.env.get_template('list.html').render(data=self.data)
    
    @cherrypy.expose
    def cell(self, cellid):
        visualize = ['dashboard',
                     'waveshapes', 
                     'timeline',
                     'spike_patterns',
                     'pattern_traces', 
                     'pattern_spike_waveforms',
                     'evoked_response']
        #methods = list(set(visualize) & set(dir(self.visualize)))
        methods = [vis for vis in visualize if vis in dir(self.visualize)]
        analyses = ["event_selector"]
        i, = np.where(self.data['id']==cellid)
        comment = self.h5filter.get_cached_string(cellid, 'comment')
            
        score = self.h5filter.get_cached_string(cellid, 'score')

        try:
            score = np.array(score)[0]
        except IndexError:
            score = "undefined"
        return self.env.get_template('cell.html').render(cellid=cellid,
                                              methods=methods,
                                              score=score,
                                              comment=comment,
                                              analyse_methods=analyses)
   
    @cherrypy.expose
    def next(self, cellid):
        i, = np.where(self.data['id']==cellid)
        try:
            newid = self.data['id'][i[0]+1]
        except IndexError:
            return "No more cells"
        cherrypy.lib.cptools.redirect("/cell?cellid=%s" % newid)

    @cherrypy.expose
    def prev(self, cellid):
        i, = np.where(self.data['id']==cellid)
        if i[0]>0:
            newid = self.data['id'][i[0]-1]
            cherrypy.lib.cptools.redirect("/cell?cellid=%s" % newid)
        return "No more cells"
    
    @cherrypy.expose
    def plot(self, method, cellid, nocache=None, clean=None,
             comment=None, reviewer=None, **kwargs):

        img_data = self.get_img_data(cellid, method, nocache, **kwargs)

        docstring = getattr(self.visualize, method).__doc__
        doc = doc2html(docstring)
         
        if reviewer: message = "You evaluation has been saved to the DB"
        else: message = ''
        
        return self.env.get_template('plot.html').render(cellid=cellid,
                                                         doc=doc,
                                                         method=method,
                                                         img_data=img_data,
                                                         message=message,
                                                         opts=kwargs)
        
from user_interface import UserInterface
data_filter = BakerlabFilter(DATAPATH+'gollum_export.inf')
abspath = os.path.abspath(os.curdir)
conf = {'/js': {'tools.staticdir.on': True,
        'tools.staticdir.dir': abspath+'/js'}}

env = Environment(loader=FileSystemLoader('templates'))

root = HelloMpl(env, data_filter)
root.analyse = UserInterface(env, data_filter)
root.visualize = Visualize(data_filter, root)
cherrypy.quickstart(root, '/', config=conf)
