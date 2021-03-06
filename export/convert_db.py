#!/usr/bin/env python
#coding=utf-8

import sys
import tables
import numpy as np
from spike_sort.io.filters import BakerlabFilter
from tables.exceptions import NoSuchNodeError
import os
import patterns
from matplotlib.mlab import rec2csv, csv2rec


#cell_db = '../cell_bwoser/src/cell_db.h5'
#DATAPATH= '/Volumes/PortableData/Data/dmytro/'

DATAPATH = os.environ['DATAPATH']
#output_fname = 'exported_cells.csv'
_, cell_db, output_fname = sys.argv


if __name__ == "__main__":
    
    records = []
    h5f = tables.openFile(cell_db)
    cell_csv = csv2rec(DATAPATH+'cell_db.csv')
    data_filter = BakerlabFilter(DATAPATH+'gollum_export.inf')

    for cellid in cell_csv['id']:
        node = h5f.getNode('/', cellid) 
        cell_path = "/".join(cellid.split('/'))
        stim_path = "/".join(cellid.split('/')[:-1]+['stim'])
        try:
            events = node.cache.events.read()
            n_events = len(events)-1
        except NoSuchNodeError:
            n_events = 1
        
        try:
            score = int(node.cache.score[0])
        except NoSuchNodeError:
            score = 0

        spt = data_filter.read_spt(cell_path)
        stim = data_filter.read_spt(stim_path)
        n_spikes = len(spt['data'])
        n_trials = len(stim['data'])
        try:
            psth, psth_time = patterns.CalcPSTH(spt['data'],
                                                stim['data'],
                                                norm=True)
            psth_peak = psth.max()
        except IndexError:
            psth_peak = np.NaN
        stim_start = 0
        stim_stop = -1
        records.append((cell_path, n_events, n_spikes, n_trials,
                        psth_peak, stim_start, stim_stop, score))

    rec_array = np.rec.fromrecords(records, 
                    names=('cell,n_events,n_spikes,n_trials,psth_height,'
                           'stim_start,stim_stop,score'))

    rec2csv(rec_array, output_fname)
