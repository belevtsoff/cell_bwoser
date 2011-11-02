#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import cherrypy

from spike_analysis import dashboard, basic
import json

from utils import doc2html

class UserInterface:
    """Select events for spike pattern analysis.

    **Usage**:

    * *Left click* -- add an event
    * *Right click* -- remove an event
    """
    
    def __init__(self, env, io_filter):

        self.io_filter = io_filter
        self.env = env
        
    @cherrypy.expose
    def event_selector(self, cellid):
        
        dataset = dashboard.read_dataset(self.io_filter, cellid)
        spt = dataset['spt']
        stim = dataset['stim']
        psth, time = basic.CalcPSTH(spt, stim)

        data = [{'x': x, 'y': y} for x, y in zip(time, psth)]
        template = self.env.get_template('event_selector.html')
       
        doc = doc2html(self.__doc__)
        
        return template.render(cellid=str(cellid),
                               data=json.dumps(data),
                               doc=doc)




