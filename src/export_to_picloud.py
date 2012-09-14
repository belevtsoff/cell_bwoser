from matplotlib import mlab
import cloud
import re
import json
import os

from contextlib import contextmanager
from tempfile import mkdtemp
import numpy as np
import tables

DATAPATH = os.environ['DATAPATH']
conf_file = os.path.join(DATAPATH, 'gollum_export.inf')
data = mlab.csv2rec(DATAPATH+'cell_db.csv')

with file(conf_file) as fid:
    conf_dict = json.load(fid)

def get_spt_filename(cellid):
    regexp = "^/(?P<subject>[a-zA-z]+)/s(?P<ses_id>.+)/el(?P<el_id>[0-9]+)/?(?P<type>[a-zA-Z]+)?(?P<cell_id>[0-9]+)?$"
    m = re.match(regexp, cellid)
    rec = m.groupdict()
    dirname = os.path.expandvars(conf_dict['dirname'])

    fspt = conf_dict[rec['type']]
    full_path = os.path.join(dirname, fspt)
    fname = full_path.format(**rec)
    return fname

@contextmanager
def get_sp_fname(nodepath, mmap='tables'):
    regexp = "^/(?P<subject>[a-zA-z]+)/s(?P<ses_id>.+)/el(?P<el_id>[0-9]+)$"
    m = re.match(regexp, nodepath)
    rec_dict = m.groupdict()

    dirname = os.path.expandvars(conf_dict['dirname'])
    n_contacts = conf_dict['n_contacts']
    f_spike = conf_dict['fspike']

    #get number of pts
    rec_dict['contact_id'] = 1
    full_path = os.path.join(dirname, f_spike)
    fname = full_path.format(**rec_dict)
    npts = os.path.getsize(fname) / 2

    dtype = 'int16'
    shape = (n_contacts, npts)

    if mmap=='numpy':
        filename = os.path.join(mkdtemp(), 'newfile.dat')
        fp = np.memmap(temp_filename, dtype=dtype, mode='w+', shape=shape)
    elif mmap=='hdf5':
        atom = tables.Atom.from_dtype(np.dtype(dtype))
        filters = tables.Filters(complevel=4, complib='zlib')
        filename = os.path.join(mkdtemp(), 'newfile.dat')
        h5f = tables.openFile(filename, 'w')
        fp = h5f.createCArray('/', "spikes", atom, shape, filters=filters)
    else:
        raise TypeError, "Bad mapping type"

    for i in range(n_contacts):
        rec_dict['contact_id'] = i + 1
        fname = full_path.format(**rec_dict)
        sp = np.fromfile(fname, dtype=np.int16)
        fp[i,:] = sp

    fp.flush()
    try:
        fp.close()
    except:
        pass

    yield filename

    os.unlink(filename)


for row in data[:2]:
    cellid = row[0]
    stimid = "/".join(cellid.split('/')[:-1]+['stim'])
    spid   = "/".join(cellid.split('/')[:-1])

    cell_fname = get_spt_filename(cellid)
    cloud.files.put(cell_fname, cellid)

    stim_fname = get_spt_filename(stimid)
    cloud.files.put(stim_fname, stimid)

    with get_sp_fname(spid, 'hdf5') as sp_fname:
        print 'File size', os.path.getsize(sp_fname)/1.E6, 'MB'
        cloud.files.put(sp_fname, spid)


