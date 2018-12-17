#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Top-level Planarian Interface for Modelling Body Organization (PLIMBO) module.
'''

# import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from collections import OrderedDict
from scipy.ndimage import rotate
from scipy.misc import imresize
import copy
import pickle
import os
import os.path
import sys, time
import csv
from betse.lib.pickle import pickles
from matplotlib import rcParams

from plimbo.harness import ModelHarness


class PlimboRunner(object):
    """
    Top level interface for running all functionality of the PLIMBO simulator.

    """

    def __init__(self, fn_config):
        """

        :param fn_config: BETSE config file for model creation and file saving information
        """

        self.fn_config = fn_config # assign path to BETSE config file


    def simRNAi(self, RNAi_vect = None, RNAi_tags = None, params = None, run_time_init = 36000.0,
                run_time_sim = 36000.0, run_time_step = 60.0, run_time_sample = 50.0,
                run_time_reinit = 12.0,  xscale = 1.0, verbose = True, new_mesh = True,
                save_dir = 'SimRNAi_1', reset_clims = True, animate = False, plot = True,
                plot_type = 'Triplot', ani_type = 'Triplot', save_harness = True, harness_type='1D'):


        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo = params, xscale = xscale, harness_type = '1D',
                                  verbose = verbose, new_mesh=new_mesh, savedir = 'Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo = params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.") # FIXME raise proper exception


        # run a single set of parameters with RNAi testing sequence:
        master.run_simRNAi(params = params, RNAi_series = RNAi_vect,
                           RNAi_names = RNAi_tags, verbose=verbose,
                           run_time_init = run_time_init,
                           run_time_sim = run_time_sim,
                           run_time_step = run_time_step,
                           run_time_sample = run_time_sample,
                           run_time_reinit = run_time_reinit,
                           reset_clims = reset_clims,
                           animate = animate,
                           plot = plot,
                           plot_type = plot_type,
                           save_dir = save_dir,
                           ani_type=ani_type)

        if save_harness is True:
            # # Save the master object: FIXME write save and load functions here, along with top-level after-plot calls
            # pickle_save_path = os.path.join(master.savepath, 'master_Search2DRNAi_1.dat')
            # pickles.save(master, filename=pickle_save_path, is_overwritable=True)

            pass

    def sensitivity(self, params=None, run_time_init=36000.0, factor = 0.25,
                run_time_sim=36000.0, run_time_step=60.0, run_time_sample=50.0,
                 xscale=1.0, verbose=True, new_mesh=True, ani_type = 'Triplot',
                save_dir='Sensitivity_1', reset_clims=True, animate=False, plot=True,
                plot_type='Triplot', save_harness=True, harness_type='1D'):

        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception


        # run a sensitivity analysis on the model:
        master.run_sensitivity(factor = factor, verbose=verbose, run_time_init = run_time_init,
                        run_time_sim = run_time_sim, run_time_step = run_time_step,
                        run_time_sample = run_time_sample, reset_clims = reset_clims, ani_type = ani_type,
                        animate = animate, plot = plot, save_dir = save_dir, plot_type = plot_type)

        if save_harness:
            pass

    def search(self, params=None, run_time_init=36000.0, factor=0.25, levels = 1, search_style = 'log',
                    run_time_sim=36000.0, run_time_step=60.0, run_time_sample=50.0,
                    xscale=1.0, verbose=True, new_mesh=True, save_dir='Search_1',
                    reset_clims=True, animate=False, plot=True, ani_type = 'Triplot',
                    plot_type='Triplot', save_harness=True, harness_type='1D'):

        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception

        # run a local search on the model, changing each parameter progressively:
        master.run_search(factor = factor, levels = levels, search_style = search_style, verbose=verbose,
                           run_time_init=run_time_init, run_time_sim=run_time_sim, run_time_step=run_time_step,
                          run_time_sample=run_time_sample, reset_clims=reset_clims, plot=plot, plot_type = plot_type,
                          animate = animate,  ani_type = ani_type, save_dir = save_dir)

        if save_harness:
            pass

    def scaleRNAi(self, params=None, xscales = None,  RNAi_vect = None, RNAi_tags = None,
                  run_time_init=36000.0, run_time_sim=36000.0, run_time_step=60.0, run_time_sample=50.0,
               run_time_reinit=12.0, xscale=1.0, verbose=True, new_mesh=True, ani_type = 'Triplot',
               save_dir='ScaleRNAi_1', reset_clims=True, animate=False, plot=True,
               plot_type='Triplot', save_harness=True, harness_type='1D'):

        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception


        # run a single set of parameters, scaling the model to various factors, with RNAi testing:
        master.run_scaleRNAi(run_params = params, xscales = xscales, RNAi_series = RNAi_vect, RNAi_names = RNAi_tags,
                               verbose=verbose, run_time_reinit=run_time_reinit, run_time_init=run_time_init,
                               run_time_sim=run_time_sim, run_time_step=run_time_step,
                             run_time_sample=run_time_sample, reset_clims=reset_clims, ani_type = ani_type,
                               plot=plot, animate=animate, save_dir=save_dir, plot_type = plot_type
                               )

        if save_harness:
            pass


    def searchRNAi(self, params=None, RNAi_vect = None, RNAi_tags = None, run_time_init = 36000.0,
                run_time_sim = 36000.0, run_time_step = 60.0, run_time_sample = 50.0, search_style = 'log',
                   factor = 0.8, levels = 1, run_time_reinit = 12.0, xscale = 1.0, verbose = True, new_mesh = True,
                save_dir = 'SimRNAi_1', reset_clims = True, animate = False, plot = True, ani_type = 'Triplot',
                plot_type = 'Triplot', save_harness = True, harness_type='1D'):

        # Create an instance of the model harness:
        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception

        # Run a local search on the model, changing each parameter progressively and running RNAi
        # set on the itteration:
        master.run_searchRNAi(RNAi_series = RNAi_vect, RNAi_names = RNAi_tags, factor = factor, levels = levels,
                                 search_style = search_style, verbose=verbose, run_time_reinit=run_time_reinit,
                                  run_time_init=run_time_init, run_time_sim=run_time_sim,
                                 run_time_step=run_time_step, run_time_sample=run_time_sample,
                                  reset_clims=reset_clims, plot=plot, plot_type = plot_type, ani_type = ani_type,
                               animate=animate, save_dir=save_dir, fixed_params = None)
        if save_harness:
            pass

    def afterplot(self):
        pass



