import importlib

from ...configurator import cfg
vis_module = importlib.import_module('_spaun.modules.vision.' + cfg.vis_module)
vis_data = vis_module.DataObject()
VisionNet = vis_module.VisionNet
