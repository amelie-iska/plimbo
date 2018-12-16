#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Abstract base classes for high-level simulator objects.
'''

from abc import ABCMeta, abstractmethod
# import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib import colorbar
# from matplotlib import rcParams
# from collections import OrderedDict
# from scipy.ndimage import rotate
# from scipy.misc import imresize
# import copy
# import pickle
# import os
# import os.path
# import sys, time
# import csv
# from betse.lib.pickle import pickles
# # from betse.util.type.mapping.mapcls import DynamicValue, DynamicValueDict
# from betse.science.parameters import Parameters


class PlanariaGRNABC(object, metaclass=ABCMeta):
    """
    Abstract base class of all objects modelling a GRN.

    A BETSE config file is used to define paths for saving image and data
    exports.
    """

    pass



class PlanariaGRN1DABC(PlanariaGRNABC):
    """
    Abstract base class of all objects modelling a one-dimensional GRN.
    """

    pass


class PlanariaGRN2DABC(PlanariaGRNABC):
    """
    Abstract base class of all objects modelling a two-dimensional GRN.
    """

    pass
