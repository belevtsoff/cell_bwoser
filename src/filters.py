import tables
import cloud
import numpy as np

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

class CloudFilter:

    def __init__(self):
        pass

    def read_spt(self,  dataset):
        """Returns spike times in miliseconds:

        Parameters
        ----------
        dataset : str
            dataset path in format
            /{subject}/session{ses_id}/el{el_id}/cell{cell_id}
        """

        cloud.files.get(dataset, 'cell.spt')

        spt = np.fromfile('cell.spt', dtype=np.int32)
        return {"data": spt / 200.0}