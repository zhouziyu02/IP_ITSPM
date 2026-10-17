# Modified: register only the required TPP model.
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_itspm import ITSPM as TorchITSPM

__all__ = ['TorchBaseModel', 'TorchITSPM']
