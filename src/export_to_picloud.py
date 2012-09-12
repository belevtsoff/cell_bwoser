from matplotlib import mlab
import cloud
import re
import json
import os

DATAPATH = os.environ['DATAPATH']
conf_file = os.path.join(DATAPATH, 'gollum_export.inf')
data = mlab.csv2rec(DATAPATH+'cell_db.csv')

with file(conf_file) as fid:
    conf_dict = json.load(fid)

def get_spt_filename(cellid):
    regexp = "^/(?P<subject>[a-zA-z]+)/s(?P<ses_id>.+)/el(?P<el_id>[0-9]+)/?(?P<type>[a-zA-Z]+)?(?P<cell_id>[0-9]+)?$"
    m = re.match(regexp, cellid)
    rec = m.groupdict()
    dirname = conf_dict['dirname'].format(**os.environ)

    fspt = conf_dict[rec['type']]
    full_path = os.path.join(dirname, fspt)
    fname = full_path.format(**rec)
    return fname

for row in data[:2]:
    cellid = row[0]
    stimid = "/".join(cellid.split('/')[:-1]+['stim'])
    cell_fname = get_spt_filename(cellid)
    cloud.files.put(cell_fname, cellid)
    stim_fname = get_spt_filename(stimid)
    cloud.files.put(stim_fname, stimid)

