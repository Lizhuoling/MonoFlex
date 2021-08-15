from .dla_dcn import bulid_dlaseg
from .vit import build_vit

def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.CONV_BODY == 'dla34':
        return bulid_dlaseg(cfg)
    elif cfg.MODEL.BACKBONE.CONV_BODY == 'vit_small':
        return build_vit(patch_size = cfg.MODEL.BACKBONE.DOWN_RATIO)
    else:
        raise Exception('Backbone {} is not surported now.'.format(cfg.MODEL.BACKBONE.CONV_BODY))