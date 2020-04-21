import zarr
import numpy as np

def to_zarr(obj,name=None,root=None):
    zz = None
    
    if type(obj)==dict:
        if root is None : zz = zarr.group()
        else            : zz = root.create_group(name)
        for key,value in obj.items() :
            to_zarr(value,name=key,root=zz) 
            
    elif type(obj) in [np.ndarray,np.float64]: root.create_dataset(name,data=obj,shape=obj.shape)
    elif type(obj)==np.int64                 : root.attrs[name] = int(obj)
    elif type(obj)== list                    : to_zarr(np.array(obj),name=name,root=root)
    elif obj is None or type(obj) in [str,bool,float,int]:root.attrs[name] = obj
    else                     : zz = to_zarr(obj.__dict__,name=name,root=root)
        
    return zz
