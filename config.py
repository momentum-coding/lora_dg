# config.py
from yacs.config import CfgNode as CN
 
def get_cfg(ds):
    if ds == 'officehome':
        _C = CN()
        _C.LORA_R = 16
        _C.LORA_LAYERS = 2
        _C.EPOCHS = 30
        _C.LR = 1e-5
        _C.BS = 32
    elif ds == 'domainnet':
        _C = CN()
        _C.LORA_R = 16
        _C.LORA_LAYERS = 0
        _C.EPOCHS = 15
        _C.LR = 5e-4
        _C.BS = 32
        
    return _C.clone()