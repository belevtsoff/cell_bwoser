import tables

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
