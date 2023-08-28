"""Extracted from https://github.com/dzyim/learn.py/blob/dev/Lib/dictionary.py."""

__all__ = ['AttrDict']

from collections.abc import Mapping

class AttrDict(Mapping):
    '''Make dictionary items as object attributes.'''
    
    force = False
    
    def __init__(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]

    def __contains__(self, key):
        return key in vars(self)
        
    def __getitem__(self, key):
        val = vars(self)[key]
        if isinstance(val, (tuple, list, set)):
            return type(val)(i.to_dict() if isinstance(i, AttrDict) else i for i in val)
        else:
            return val.to_dict() if isinstance(val, AttrDict) else val
    
    def __iter__(self):
        return iter(vars(self))
    
    def __len__(self):
        return len(vars(self))
    
    def __repr__(self):
        L = ('%s: %s' % (repr(k), repr(v)) for k, v in self.items())
        return 'AttrDict({%s})' % ', '.join(L)
    
    def __setitem__(self, key, val):
        if isinstance(val, (tuple, list, set)):
            L = type(val)(AttrDict.from_dict(i, AttrDict.force) if isinstance(i, dict) else i for i in val)
            setattr(self, key, L)
        else:
            setattr(self, key, AttrDict.from_dict(val, AttrDict.force) if isinstance(val, dict) else val)
    
    @classmethod
    def from_dict(cls, dct, force=False):
        cls.force = force
        if cls.force:
            dct = {str(key): dct[key] for key in dct}
        new = cls(**dct)
        cls.force = False
        return new
    
    def to_dict(self):
        return dict((key, self[key]) for key in self)
    
    fromdict = fromDict = from_dict
    todict = toDict = to_dict
    
