import importlib

from ...configurator import cfg
mtr_module = importlib.import_module('_spaun.modules.motor.' + cfg.mtr_module)
mtr_data = mtr_module.DataObject()
Controller = mtr_module.Controller

from .sig_ramp_net import Ramp_Signal_Network
