import importlib

from ...configurator import cfg
stim_module = importlib.import_module('_spaun.modules.stim.' + cfg.stim_module)
stim_data = stim_module.DataObject()
