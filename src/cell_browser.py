import cherrypy
from jinja2 import FileSystemLoader, Environment

from filters import HDF5filter, CloudFilter
from plotting import Visualize
from matplotlib import mlab

import os
import cloud

DATAPATH = os.environ['DATAPATH']

import json
import numpy as np

from utils import doc2html

#cloud.start_simulator()

def append_column(data, type):
    dtype = data.dtype.descr
    dtype.append(('score', np.int8))
    new_data = np.empty(data.shape, dtype=dtype)

    for name, type in data.dtype.descr:
        new_data[name] = data[name]
    return new_data

def picloud_task(method, cellid, kwargs):
    visualize = Visualize(CloudFilter())
    getattr(visualize, method)(cellid, **kwargs)
    img_data = visualize.html_fig()
    return img_data


class HelloMpl:
    def __init__(self, env):

        self.env = env
        
        self.h5filter = HDF5filter('cell_db.h5')
        
        self.data = mlab.csv2rec(DATAPATH+'cell_db.csv')
        self.data = append_column(self.data, ('score', np.int8))
        self.running_tasks = {}
   
    def update_score(self):

        for i in range(len(self.data)):
            cell_id = self.data['id'][i]
            score = self.h5filter.get_cached_string(cell_id, 'score')
            if score:
                self.data['score'][i] = int(score[0])
            else:
                self.data['score'][i] = 0 

    def get_img_data(self, cellid, method, opts, nocache=None):


        item = 'img_'+method


        if (cellid, item) in self.running_tasks:
            jid = self.running_tasks[(cellid, item)]
            status = cloud.status(jid)
            if status in ['processing', 'queued']:
                return 'inprogress', None
            elif status=='done':
                img_data = cloud.result(status)
                self.h5filter.add_cached_string(cellid, item, img_data)
                del self.running_tasks[(cellid, item)]
                return 'done', img_data
            else:
                del self.running_tasks[(cellid, item)]
                import pdb; pdb.set_trace()
                return 'error', None

        img_data = self.h5filter.get_cached_string(cellid, item)
        
        if not img_data or nocache or opts:
            jid = cloud.call(picloud_task, method, cellid, opts, _env='spikesort', _profile=True, _type='c2')
            self.running_tasks[(cellid, item)] = jid
            return 'inprogress', None

        return 'done', img_data
    
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
        self.update_score()
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
        methods = [vis for vis in visualize]
        analyses = ["event_selector"] 
        return self.env.get_template('cell.html').render(cellid=cellid,
                                              methods=methods,
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
    def img(self, cellid, method, opts=None):
        if not opts:
            opts = {}
        else:
            opts = json.loads(opts)

        status, img_data = self.get_img_data(cellid, method, opts)
        return json.dumps({'status': status, 'data': img_data})

    @cherrypy.expose
    def plot(self, method, cellid, nocache=None, clean=None,
             comment=None, reviewer=None, **kwargs):

        docstring = getattr(Visualize, method).__doc__
        doc = doc2html(docstring)
         
        if reviewer: message = "You evaluation has been saved to the DB"
        else: message = ''
        
        return self.env.get_template('plot.html').render(cellid=cellid,
                                                         doc=doc,
                                                         method=method,
                                                         message=message,
                                                         opts=kwargs)
        
from user_interface import UserInterface

abspath = os.path.abspath(os.curdir)
conf = {'/js': {'tools.staticdir.on': True,
        'tools.staticdir.dir': abspath+'/js'}}

env = Environment(loader=FileSystemLoader('templates'))

root = HelloMpl(env)
root.analyse = UserInterface(env, CloudFilter())

cherrypy.quickstart(root, '/', config=conf)
