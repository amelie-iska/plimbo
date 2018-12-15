#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''

Harness for running automated model searches and sensitivity analyses for the 1D and 2D simulators of the
PLIMBO module.

'''

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
from betse.util.path import dirs, pathnames
from matplotlib import rcParams

from plimbo.sim1D import PlanariaGRN1D
from plimbo.sim2D import PlanariaGRN2D
from plimbo.auto_params import ParamsManager


class ModelHarness(object):

    def __init__(self, config_filename, paramo, xscale=1.0, harness_type = '1D',
                 verbose = False, new_mesh=False, savedir = 'ModelSearch'):

        self.xscale = xscale

        self.paramo = paramo

        self.verbose = verbose

        self.new_mesh = new_mesh

        self.savedir = savedir

        self.config_fn = config_filename

        # Create a simulator object:
        if harness_type == '1D':
            self.model = PlanariaGRN1D(config_filename, self.paramo, xscale=1.0, verbose=False, new_mesh=False)

        if harness_type == '2D':
            self.model = PlanariaGRN2D(config_filename, self.paramo, xscale=1.0, verbose=False, new_mesh=False)

        # save the harness type so we know what we're working with:
        self.harness_type = harness_type

        # Create a directory to store results:
        self.savepath = pathnames.join_and_canonicalize(self.model.p.conf_dirname, savedir)

        os.makedirs(self.savepath, exist_ok=True)

        # Create a dictionary to save sub-folder paths:
        self.subfolders_dict = OrderedDict()

        # Generate a parameters manager object for the harness:
        self.pm = ParamsManager(self.paramo)

        # initialize outputs array to null:
        self.outputs = []

        # save the model's concentration tags:
        self.conctags = self.model.conc_tags


    def run_sensitivity(self, factor = 0.1, verbose=True, run_time_init = 36000.0,
                        run_time_sim = 36000.0, run_time_step = 60, run_time_sample = 50,
                        reset_clims = True, animate = False, plot = True, save_dir = 'Sensitivity1'):

        # general saving directory for this procedure:
        self.savedir_sensitivity = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_sensitivity, exist_ok=True)

        self.subfolders_dict['sensitivity'] = self.savedir_sensitivity

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        self.pm.create_sensitivity_matrix(factor=factor)  # Generate sensitivity matrix from default parameters

        self.outputs = []  # Storage array for all data created in each itteration of the model

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
            data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim

            if verbose is True:
                print('Run ', ii + 1, " of ", self.pm.N_runs)

            # convert the array to a dictionary:
            run_params = OrderedDict(zip(self.pm.param_labels, params_list))

            # create a model using the specific parameters from the params manager for this run:
            self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                verbose=self.verbose, new_mesh=self.new_mesh)

            # Run initialization of full model:
            self.model.initialize(knockdown= None,
                                   run_time=run_time_init,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)

            self.model.simulate(knockdown= None,
                                   run_time=run_time_sim,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)


            data_dict_inits['base'] = self.model.molecules_time.copy()
            data_dict_sims['base'] = self.model.molecules_sim_time.copy()

            self.outputs.append([data_dict_inits, data_dict_sims])

            if plot:
                self.plot_single('base', ii, harness_type='sensitivity', plot_type='Triplot', output_type='init')
                self.plot_single('base', ii, harness_type='sensitivity', plot_type='Triplot', output_type='sim')

            if animate:
                self.ani_single('base', ii, harness_type='sensitivity', ani_type='Triplot', output_type='init')
                self.ani_single('base', ii, harness_type='sensitivity', ani_type='Triplot', output_type='sim')

            if verbose is True:
                print('----------------')


    def run_search(self, factor = 0.8, levels = 1, search_style = 'log', verbose=True,
                   run_time_init=36000.0, run_time_sim=36000.0, run_time_step=60, run_time_sample=50,
                   reset_clims=True, plot=True, animate = False, save_dir = 'Search1',
                   fixed_params = None):

        # general saving directory for this procedure:
        self.savedir_search = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_search, exist_ok=True)

        self.subfolders_dict['search'] = self.savedir_search

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        # Create the parameters matrix:
        self.pm.create_search_matrix(factor=factor, levels=levels, style=search_style)

        self.outputs = []  # Storage array for all last timestep outputs

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            if verbose is True:
                print('Run ', ii + 1, " of ", self.pm.N_runs)

            data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
            data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim

            # convert the array to a dictionary:
            run_params = OrderedDict(zip(self.pm.param_labels, params_list))

            # create a model using the specific parameters from the params manager for this run:
            self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                verbose=self.verbose, new_mesh=self.new_mesh)

            # Run initialization of full model:
            self.model.initialize(knockdown= None,
                                   run_time=run_time_init,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)

            self.model.simulate(knockdown= None,
                                   run_time=run_time_sim,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)


            data_dict_inits['base'] = self.model.molecules_time.copy()
            data_dict_sims['base'] = self.model.molecules_sim_time.copy()

            self.outputs.append([data_dict_inits, data_dict_sims])

            if plot:
                self.plot_single('base', ii, harness_type='search', plot_type='Triplot', output_type='init')
                self.plot_single('base', ii, harness_type='search', plot_type='Triplot', output_type='sim')

            if animate:
                self.ani_single('base', ii, harness_type='search', ani_type='Triplot', output_type='init')
                self.ani_single('base', ii, harness_type='search', ani_type='Triplot', output_type='sim')

            if verbose is True:
                print('----------------')

    def run_searchRNAi(self, RNAi_series, RNAi_names, factor = 0.8, levels = 1, search_style = 'log',
                        verbose=True, run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True,
                       animate=False, save_dir='SearchRNAi1', fixed_params = None):

        # general saving directory for this procedure:
        self.savedir_searchRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_searchRNAi, exist_ok=True)

        self.subfolders_dict['searchRNAi'] = self.savedir_searchRNAi

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        # add in new datatags for RNAi_series
        for rnai_n in RNAi_names:
            self.datatags.append(rnai_n)

        # Create the parameters matrix:
        self.pm.create_search_matrix(factor=factor, levels = levels, style = search_style)

        self.outputs = []  # Storage array for all last timestep outputs

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            if verbose is True:
                print('Run ', ii + 1, " of ", self.pm.N_runs)

            data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
            data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim

            # convert the array to a dictionary:
            run_params = OrderedDict(zip(self.pm.param_labels, params_list))

            # create a model using the specific parameters from the params manager for this run:
            self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                verbose=self.verbose, new_mesh=self.new_mesh)

            # Run initialization of full model:
            self.model.initialize(knockdown= None,
                                   run_time=run_time_init,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)

            self.model.simulate(knockdown= None,
                                   run_time=run_time_sim,
                                   run_time_step=run_time_step,
                                   run_time_sample=run_time_sample,
                                   reset_clims = reset_clims)


            data_dict_inits['base'] = self.model.molecules_time.copy()
            data_dict_sims['base'] = self.model.molecules_sim_time.copy()

            self.outputs.append([data_dict_inits, data_dict_sims])

            if plot:
                self.plot_single('base', ii, harness_type='searchRNAi', plot_type='Triplot', output_type='init')
                self.plot_single('base', ii, harness_type='searchRNAi', plot_type='Triplot', output_type='sim')

            if animate:
                self.ani_single('base', ii, harness_type='searchRNAi', ani_type='Triplot', output_type='init')
                self.ani_single('base', ii, harness_type='searchRNAi', ani_type='Triplot', output_type='sim')

            if verbose is True:
                print('----------------')

            for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                if verbose is True:
                    print('Runing RNAi Sequence ', rnai_n)

                # Reinitialize the model again:
                self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                      verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization phase of full model:
                self.model.initialize(knockdown=None,
                                      run_time=run_time_init,
                                      run_time_step=run_time_step,
                                      run_time_sample=run_time_sample,
                                      reset_clims=reset_clims)

                if run_time_reinit > 0.0:  # if there is a reinitialization phase (RNAi applied, no cutting)
                    self.model.reinitialize(knockdown=rnai_s,
                                            run_time=run_time_reinit,
                                            run_time_step=run_time_step,
                                            run_time_sample=run_time_sample
                                            )
                # Run the simulation with RNAi intervention applied:
                self.model.simulate(knockdown=rnai_s,
                                    run_time=run_time_sim,
                                    run_time_step=run_time_step,
                                    run_time_sample=run_time_sample,
                                    reset_clims=False)

                # Save whole molecules master arrays to their respective data dictionaries:
                data_dict_inits[rnai_n] = self.model.molecules_time.copy()
                data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

                if plot:
                    self.plot_single(rnai_n, ii, harness_type='searchRNAi', plot_type='Triplot', output_type='sim')

                if animate:
                    self.ani_single(rnai_n, ii, harness_type='searchRNAi', ani_type='Triplot', output_type='sim')

                if verbose is True:
                    print('----------------')

    def run_scaleRNAi(self, run_params, xscales, RNAi_series, RNAi_names, verbose=True,
                       run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True,
                       animate=False, save_dir='scaleRNAi1'
                       ):

        # general saving directory for this procedure:
        self.savedir_scaleRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_scaleRNAi, exist_ok=True)

        self.subfolders_dict['scaleRNAi'] = self.savedir_scaleRNAi

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        # add in new datatags for RNAi_series
        for rnai_n in RNAi_names:
            self.datatags.append(rnai_n)

        self.outputs = []  # Storage array for all last timestep outputs

        for ii, xxs in enumerate(xscales):  # Step through the x-scaling factors

            if verbose is True:
                print('Run ', ii + 1, " of ", len(xscales))

            data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
            data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim

            # create a model using the specific parameters from the params manager for this run at this scale:
            self.model.model_init(self.config_fn, run_params, xscale=xxs,
                                  verbose=self.verbose, new_mesh=self.new_mesh)

            # Run initialization of full model:
            self.model.initialize(knockdown=None,
                                  run_time=run_time_init,
                                  run_time_step=run_time_step,
                                  run_time_sample=run_time_sample,
                                  reset_clims=reset_clims)

            self.model.simulate(knockdown=None,
                                run_time=run_time_sim,
                                run_time_step=run_time_step,
                                run_time_sample=run_time_sample,
                                reset_clims=reset_clims)

            data_dict_inits['base'] = self.model.molecules_time.copy()
            data_dict_sims['base'] = self.model.molecules_sim_time.copy()

            self.outputs.append([data_dict_inits, data_dict_sims])

            if plot:
                self.plot_single('base', ii, harness_type='scaleRNAi', plot_type='Triplot', output_type='init')
                self.plot_single('base', ii, harness_type='scaleRNAi', plot_type='Triplot', output_type='sim')

            if animate:
                self.ani_single('base', ii, harness_type='scaleRNAi', ani_type='Triplot', output_type='init')
                self.ani_single('base', ii, harness_type='scaleRNAi', ani_type='Triplot', output_type='sim')

            if verbose is True:
                print('----------------')

            for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                if verbose is True:
                    print('Runing RNAi Sequence ', rnai_n)

                # Reinitialize the model again:
                self.model.model_init(self.config_fn, run_params, xscale=xxs,
                                      verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization phase of full model:
                self.model.initialize(knockdown=None,
                                      run_time=run_time_init,
                                      run_time_step=run_time_step,
                                      run_time_sample=run_time_sample,
                                      reset_clims=reset_clims)

                if run_time_reinit > 0.0:  # if there is a reinitialization phase (RNAi applied, no cutting)
                    self.model.reinitialize(knockdown=rnai_s,
                                            run_time=run_time_reinit,
                                            run_time_step=run_time_step,
                                            run_time_sample=run_time_sample
                                            )
                # Run the simulation with RNAi intervention applied:
                self.model.simulate(knockdown=rnai_s,
                                    run_time=run_time_sim,
                                    run_time_step=run_time_step,
                                    run_time_sample=run_time_sample,
                                    reset_clims=False)

                # Save whole molecules master arrays to their respective data dictionaries:
                data_dict_inits[rnai_n] = self.model.molecules_time.copy()
                data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

                if plot:
                    self.plot_single(rnai_n, ii, harness_type='scaleRNAi', plot_type='Triplot', output_type='sim')

                if animate:
                    self.ani_single(rnai_n, ii, harness_type='scaleRNAi', ani_type='Triplot', output_type='sim')

                if verbose is True:
                    print('----------------')

    def run_simRNAi(self, run_params, RNAi_series, RNAi_names, verbose=True, run_time_init = 36000.0,
                 run_time_sim = 36000.0, run_time_step = 60, run_time_sample = 50, run_time_reinit = 12,
                 reset_clims = True, animate = False, plot = True, save_dir = 'SimRNAi_1'):


        # general saving directory for this procedure:
        self.savedir_simRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_simRNAi, exist_ok=True)

        self.subfolders_dict['simRNAi'] = self.savedir_simRNAi


        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        # add in new datatags for RNAi_series
        for rnai_n in RNAi_names:
            data_tag = rnai_n
            self.datatags.append(data_tag)

        self.outputs = []  # Storage array for outputs of each model itteration

        data_dict_inits = OrderedDict() # storage of inits for each molecules of a model itterantion
        data_dict_sims = OrderedDict() # storage of sims for each molecules of a model itterantion

        # create a model using the specific parameters from the params manager for this run:
        self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                              verbose=self.verbose, new_mesh=self.new_mesh)

        # Run initialization of full model:
        self.model.initialize(knockdown=None,
                              run_time=run_time_init,
                              run_time_step=run_time_step,
                              run_time_sample=run_time_sample,
                              reset_clims=reset_clims)

        # Run simulation of full model:
        self.model.simulate(knockdown=None,
                              run_time=run_time_sim,
                              run_time_step=run_time_step,
                              run_time_sample=run_time_sample,
                              reset_clims=reset_clims)

        # Save whole molecules master arrays to their respective data dictionaries:
        data_dict_inits['base'] = self.model.molecules_time.copy()
        data_dict_sims['base'] = self.model.molecules_sim_time.copy()

        if plot:
            self.plot_single('base', 0, harness_type='simRNAi', plot_type='Triplot', output_type='init')
            self.plot_single('base', 0, harness_type='simRNAi', plot_type='Triplot', output_type='sim')

        if animate:
            self.ani_single('base', 0, harness_type='simRNAi', ani_type='Triplot', output_type='init')
            self.ani_single('base', 0, harness_type='simRNAi', ani_type='Triplot', output_type='sim')

        if verbose is True:
            print('----------------')

        for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

            if verbose is True:
                print('Runing RNAi Sequence ', rnai_n)

            # Reinitialize the model again:
            self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                  verbose=self.verbose, new_mesh=self.new_mesh)

            # Run initialization phase of full model:
            self.model.initialize(knockdown=None,
                                  run_time=run_time_init,
                                  run_time_step=run_time_step,
                                  run_time_sample=run_time_sample,
                                  reset_clims=reset_clims)

            if run_time_reinit > 0.0: # if there is a reinitialization phase (RNAi applied, no cutting)
                self.model.reinitialize(knockdown=rnai_s,
                                    run_time=run_time_reinit,
                                    run_time_step=run_time_step,
                                    run_time_sample=run_time_sample
                                   )
            # Run the simulation with RNAi intervention applied:
            self.model.simulate(knockdown=rnai_s,
                            run_time=run_time_sim,
                            run_time_step=run_time_step,
                            run_time_sample=run_time_sample,
                            reset_clims=False)

            # Save whole molecules master arrays to their respective data dictionaries:
            data_dict_inits[rnai_n] = self.model.molecules_time.copy()
            data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

            if plot:
                self.plot_single(rnai_n, 0, harness_type='simRNAi', plot_type='Triplot', output_type='sim')


            if animate:
                self.ani_single(rnai_n, 0, harness_type='simRNAi', ani_type='Triplot', output_type='sim')

            if verbose is True:
                print('----------------')



        self.outputs.append([data_dict_inits, data_dict_sims])



    def plot_all_output(self, harness_type=None, plot_type='Triplot', output_type='sim'):

        if harness_type is None:
            harness_type = ''
            plotdirmain = self.savepath

        else:
            plotdirmain = self.subfolders_dict[harness_type]

        if self.verbose is True:
            print('Plotting ', harness_type, '...')

        # Reinitialize the model again:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                              verbose=False, new_mesh=False)

        if self.harness_type == '2D' and output_type == 'sim':
            self.model.cut_cells()

        if len(self.outputs):

            if output_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(self.outputs):

                    for init_i, tagi in zip(inits_dict.values(), self.datatags):

                        if self.verbose is True:
                            print('Plotting init of', tagi, 'for run ', ri, '...')

                        fni = plot_type + '_' + tagi + '_init_' + str(ri)

                        self.model.molecules_time = init_i

                        if plot_type == 'Triplot':
                            self.model.triplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                               cmaps=None, clims=None, autoscale=False)

                        elif plot_type == 'Biplot':
                            self.model.biplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                               cmaps=None, clims=None, autoscale=False)

            elif output_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(self.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), self.datatags):

                        if self.verbose is True:
                            print('Plotting sim of', tagi, 'for run ', ri, '...')

                        fns = plot_type + '_' + tagi + '_sim_' + str(ri)
                        self.model.molecules_sim_time = sim_i

                        if plot_type == 'Triplot':
                            self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                               cmaps=None, clims=None, autoscale=False)

                        elif plot_type == 'Biplot':
                            self.model.biplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                              cmaps=None, clims=None, autoscale=False)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to plot.')

    def ani_all_output(self, harness_type=None, ani_type='Triplot', output_type='sim'):

        if harness_type is None:
            harness_type = ''
            plotdirmain = self.savepath

        else:
            plotdirmain = self.subfolders_dict[harness_type]

        if self.verbose is True:
            print('Plotting ', harness_type, '...')

        # Reinitialize the model again:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                              verbose=False, new_mesh=False)

        if self.harness_type == '2D' and output_type == 'sim':
            self.model.cut_cells()

        if len(self.outputs):

            if output_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(self.outputs):

                    for init_i, tagi in zip(inits_dict.values(), self.datatags):

                        if self.verbose is True:
                            print('Animating init of', tagi, 'for run ', ri, '...')

                        dni = ani_type + '_' + tagi + '_init'+ str(ri)
                        plotdiri = os.path.join(plotdirmain, dni)

                        self.model.molecules_time = init_i

                        if ani_type == 'Triplot':
                            self.model.animate_triplot(ani_type='init', dirsave=plotdiri,
                                               cmaps=None, clims=None, autoscale=False)

                        elif ani_type == 'Biplot':
                            self.model.animate_biplot(ani_type='init', dirsave=plotdiri,
                                              cmaps=None, clims=None, autoscale=False)

            elif output_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(self.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), self.datatags):

                        if self.verbose is True:
                            print('Plotting sim of', tagi, 'for run ', ri, '...')

                        dns = ani_type + '_' + tagi + '_sim' + str(ri)
                        plotdirs =os.path.join(plotdirmain, dns)

                        self.model.molecules_sim_time = sim_i

                        if ani_type == 'Triplot':
                            self.model.animate_triplot(ani_type='sim', dirsave=plotdirs,
                                               cmaps=None, clims=None, autoscale=False)

                        elif ani_type == 'Biplot':
                            self.model.animate_biplot(ani_type='sim', dirsave=plotdirs,
                                              cmaps=None, clims=None, autoscale=False)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to animate.')

    def plot_single(self, tagi, ri, harness_type=None, plot_type='Triplot', output_type='sim'):
        """

        :param tagi: datatag for the plot
        :param ri: itteration index of the model
        :param harness_type:  type of harness being simulated
        :param plot_type: plot a 'Triplot' or a 'Biplot'
        :param output_type:  plot an 'init' or a 'sim'
        :return: None
        """

        if harness_type is None:
            harness_type = ''
            plotdirmain = self.savepath

        else:
            plotdirmain = self.subfolders_dict[harness_type]

        if output_type == 'init':

            if self.verbose is True:
                print('Plotting init of', tagi, 'for run ', ri, '...')

            fni = plot_type + '_' + tagi + '_init_' + str(ri)

            if plot_type == 'Triplot':
                self.model.triplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False)

            elif plot_type == 'Biplot':
                self.model.biplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False)

        elif output_type == 'sim':

            if self.verbose is True:
                print('Plotting sim of', tagi, 'for run ', ri, '...')

            fns = plot_type + '_' + tagi + '_sim_' + str(ri)

            if plot_type == 'Triplot':
                self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False)

            elif plot_type == 'Biplot':
                self.model.biplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                  cmaps=None, clims=None, autoscale=False)

    def ani_single(self, tagi, ri, harness_type=None, ani_type='Triplot', output_type='sim'):

        if harness_type is None:
            harness_type = ''
            plotdirmain = self.savepath

        else:
            plotdirmain = self.subfolders_dict[harness_type]

        if output_type == 'init':

            if self.verbose is True:
                print('Animating init of', tagi, 'for run ', ri, '...')

            dni = ani_type + '_' + tagi + '_init'+ str(ri)
            plotdiri = os.path.join(plotdirmain, dni)

            if ani_type == 'Triplot':
                self.model.animate_triplot(ani_type='init', dirsave=plotdiri,
                                   cmaps=None, clims=None, autoscale=False)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='init', dirsave=plotdiri,
                                  cmaps=None, clims=None, autoscale=False)

        elif output_type == 'sim':

            if self.verbose is True:
                print('Animating sim of', tagi, 'for run ', ri, '...')

            dns = ani_type + '_' + tagi + '_sim'+ str(ri)
            plotdirs =os.path.join(plotdirmain, dns)

            if ani_type == 'Triplot':
                self.model.animate_triplot(ani_type='sim', dirsave=plotdirs,
                                   cmaps=None, clims=None, autoscale=False)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='sim', dirsave=plotdirs,
                                  cmaps=None, clims=None, autoscale=False)




    # def lineplot_1D(self, conc, fname, runtype='init'):
    #
    #     if runtype == 'init':
    #         plt.figure()
    #         plt.plot(self.X * 1e3, conc, '-r', linewidth=2.0)
    #         plt.xlabel('Distance [mm]')
    #         plt.ylabel('Concentration [mM/L]')
    #         plt.savefig(fname, format='png', dpi=300)
    #         plt.close()
    #
    #     if runtype == 'sim':
    #
    #         coo = conc
    #         xoo = self.X * 1e3
    #
    #         sim_coo = []
    #         sim_xoo = []
    #
    #         for a, b in self.seg_inds:
    #             sim_coo.append(coo[a:b])
    #             sim_xoo.append(xoo[a:b])
    #
    #         sim_coo = np.asarray(sim_coo)
    #         sim_xoo = np.asarray(sim_xoo)
    #
    #         plt.figure()
    #
    #         for xi, ci in zip(sim_xoo, sim_coo):
    #             plt.plot(xi, ci, color='red', linewidth=2.0)
    #
    #         plt.xlabel('Distance [mm]')
    #         plt.ylabel('Concentration [mM/L]')
    #         plt.savefig(fname, format='png', dpi=300)
    #         plt.close()
    #
    # def plot_indy(self, ii, dataname):
    #     """
    #
    #     Plot an individual parameters run.
    #
    #     """
    #
    #     ci = self.outputs[0][dataname]
    #     cj = self.outputs[ii][dataname]
    #
    #     impath = os.path.join(self.savepath, dataname)
    #     os.makedirs(impath, exist_ok=True)
    #
    #     fnme = dataname + str(ii) + '.png'
    #
    #     impath_c_init = os.path.join(impath, fnme)
    #
    #     # calculate percent changes to input variables in this run
    #     percent_delta_input = 100 * ((self.p.params_M[ii, :] -
    #                                   self.p.params_M[0, :]) / self.p.params_M[0, :])
    #
    #     index_delta = (percent_delta_input != 0.0).nonzero()[0]
    #
    #     delta_i = np.round(percent_delta_input[index_delta], 1)
    #     name_i = self.p.param_labels[index_delta]
    #
    #     me = np.round((1 / len(self.Xscale)) * (100 * ((cj - ci) / ci)).sum(), 2)
    #
    #     msg_o = 'ΔOutput ' + '  ' + str(me) + '%'
    #
    #     if ii > 0:
    #
    #         plt.figure()
    #         ax = plt.subplot(111)
    #         plt.plot(self.Xscale * 1e3, cj, '-r', linewidth=2.0, label='Altered')
    #         plt.plot(self.Xscale * 1e3, ci, '--b', linewidth=1.0, label='Original')
    #         plt.xlabel('Scaled Distance')
    #         plt.ylabel('Concentration [mM/L]')
    #
    #         # add text to plot describing which parameters were changed in this run,
    #         # and by how much:
    #         if len(delta_i):
    #
    #             for si, (ni, di) in enumerate(zip(name_i, delta_i)):
    #                 msg_i = 'Δ' + ni + '  ' + str(di) + '%'
    #
    #                 plt.text(0.02, 0.95 - (si / 15), msg_i, transform=ax.transAxes)
    #
    #                 #                 Add text describing how output changes for this run:
    #         plt.text(0.02, 0.95 - (len(delta_i) / 15), msg_o,
    #                  transform=ax.transAxes)
    #
    #         plt.savefig(impath_c_init, format='png', dpi=300)
    #         plt.close()
    #
    # # Plot Results
    # def plot_data(self, dataname):
    #
    #     change_input_msgs, change_output_msg, _, _ = self.work_changes(dataname)
    #
    #     for ii in range(self.p.N_runs):
    #
    #         ci = self.outputs[0][dataname]
    #         cj = self.outputs[ii][dataname]
    #
    #         impath = os.path.join(self.savepath, dataname)
    #         os.makedirs(impath, exist_ok=True)
    #
    #         fnme = dataname + str(ii) + '.png'
    #
    #         impath_c_init = os.path.join(impath, fnme)
    #
    #         if ii > 0:
    #
    #             plt.figure()
    #             ax = plt.subplot(111)
    #             plt.plot(self.Xs * 1e3, cj, '-r', linewidth=2.0, label='Altered')
    #             plt.plot(self.Xs * 1e3, ci, '--b', linewidth=1.0, label='Original')
    #             plt.xlabel('Scaled Distance')
    #             plt.ylabel('Concentration [mM/L]')
    #
    #             # add text to plot describing which parameters were changed in this run,
    #             # and by how much:
    #             for si, msgi in enumerate(change_input_msgs[ii]):
    #                 plt.text(0.02, 0.95 - (si / 15), msgi, transform=ax.transAxes)
    #
    #             # Add text describing how output changes for this run:
    #             plt.text(0.02, 0.95 - (len(change_input_msgs[ii]) / 15), change_output_msg[ii],
    #                      transform=ax.transAxes)
    #
    #             plt.savefig(impath_c_init, format='png', dpi=300)
    #             plt.close()
    #
    # def output_delta_table(self, dataname):
    #
    #     _, _, self.change_input, self.change_output = self.work_changes(dataname)
    #
    #     hdr = ''
    #
    #     for plab in self.p.param_labels:
    #         hdr += '% Δ' + plab + ','
    #     hdr += '% ΔOutput'
    #
    #     writeM = np.column_stack((self.change_input[1:], self.change_output[1:]))
    #
    #     fnme = dataname + '_analysis.csv'
    #
    #     fpath = os.path.join(self.savepath, fnme)
    #
    #     np.savetxt(fpath, writeM, delimiter=',', header=hdr)
    #
    # def output_summary_table(self, dataname):
    #
    #     _, _, self.change_input, self.change_output = self.work_changes(dataname)
    #
    #     a, b = self.change_input[1:].shape
    #
    #     if a == b:
    #         diag_vals = np.diag(self.change_input[1:])
    #
    #         hdr = 'Parameter, %Δ Parameter, %Δ Output'
    #
    #         writeM = np.column_stack((self.p.param_labels, diag_vals, self.change_output[1:]))
    #
    #         fnme = dataname + '_sensitiviy_analysis.csv'
    #
    #         fpath = os.path.join(self.savepath, fnme)
    #
    #         np.savetxt(fpath, writeM, fmt='%s', delimiter=',', header=hdr)
    #
    #     head = ''
    #     for pi in self.p.param_labels:
    #         head += pi + ','
    #
    #     fpath2 = os.path.join(self.savepath, 'param_values.csv')
    #     np.savetxt(fpath2, self.p.params_M, delimiter=',', header=head)
    #
    # def work_changes(self, dataname):
    #
    #     msg_inputs = []
    #     msg_outputs = []
    #
    #     tab_inputs = []
    #     tab_outputs = []
    #
    #     for ii in range(self.p.N_runs):
    #
    #         msg_in = []
    #
    #         ci = self.outputs[0][dataname]
    #         cj = self.outputs[ii][dataname]
    #
    #         # calculate percent changes to input variables in this run
    #         percent_delta_input = 100 * ((self.p.params_M[ii, :] -
    #                                       self.p.params_M[0, :]) / self.p.params_M[0, :])
    #
    #         index_delta = (percent_delta_input != 0.0).nonzero()[0]
    #
    #         delta_i = np.round(percent_delta_input[index_delta], 1)
    #         name_i = self.p.param_labels[index_delta]
    #
    #         for si, (ni, di) in enumerate(zip(name_i, delta_i)):
    #             msg = 'Δ' + ni + '  ' + str(di) + '%'
    #
    #             msg_in.append(msg)
    #
    #         me = np.round((1 / len(self.X)) * (100 * ((cj - ci) / ci)).sum(), 2)
    #
    #         msg_o = 'ΔOutput ' + '  ' + str(me) + '%'
    #
    #         msg_outputs.append(msg_o)
    #
    #         msg_inputs.append(msg_in)
    #
    #         tab_outputs.append(me * 1)
    #         tab_inputs.append(percent_delta_input * 1)
    #
    #     tab_inputs = np.asarray(tab_inputs)
    #     tab_outputs = np.asarray(tab_outputs)
    #
    #     return msg_inputs, msg_outputs, tab_inputs, tab_outputs
    #
    # def render_output(self, makeplots=True):
    #
    #     for tag in self.datatags:  # Save plots of each concentration to disk
    #
    #         if makeplots:
    #             self.plot_data(tag)
    #
    #         # Save spreadsheet of model changes for parameter changes:
    #         self.output_delta_table(tag)
    #
    #         self.output_summary_table(tag)




    # def lineplot_1D(self, conc, fname, runtype='init'):
    #
    #     if runtype == 'init':
    #         plt.figure()
    #         plt.plot(self.X * 1e3, conc, '-r', linewidth=2.0)
    #         plt.xlabel('Distance [mm]')
    #         plt.ylabel('Concentration [mM/L]')
    #         plt.savefig(fname, format='png', dpi=300)
    #         plt.close()
    #
    #     if runtype == 'sim':
    #
    #         coo = conc
    #         xoo = self.X * 1e3
    #
    #         sim_coo = []
    #         sim_xoo = []
    #
    #         for a, b in self.seg_inds:
    #             sim_coo.append(coo[a:b])
    #             sim_xoo.append(xoo[a:b])
    #
    #         sim_coo = np.asarray(sim_coo)
    #         sim_xoo = np.asarray(sim_xoo)
    #
    #         plt.figure()
    #
    #         for xi, ci in zip(sim_xoo, sim_coo):
    #             plt.plot(xi, ci, color='red', linewidth=2.0)
    #
    #         plt.xlabel('Distance [mm]')
    #         plt.ylabel('Concentration [mM/L]')
    #         plt.savefig(fname, format='png', dpi=300)
    #         plt.close()
    #
    # def plot_indy(self, ii, dataname):
    #     """
    #
    #     Plot an individual parameters run.
    #
    #     """
    #
    #     ci = self.outputs[0][dataname]
    #     cj = self.outputs[ii][dataname]
    #
    #     impath = os.path.join(self.savepath, dataname)
    #     os.makedirs(impath, exist_ok=True)
    #
    #     fnme = dataname + str(ii) + '.png'
    #
    #     impath_c_init = os.path.join(impath, fnme)
    #
    #     # calculate percent changes to input variables in this run
    #     percent_delta_input = 100 * ((self.p.params_M[ii, :] -
    #                                   self.p.params_M[0, :]) / self.p.params_M[0, :])
    #
    #     index_delta = (percent_delta_input != 0.0).nonzero()[0]
    #
    #     delta_i = np.round(percent_delta_input[index_delta], 1)
    #     name_i = self.p.param_labels[index_delta]
    #
    #     me = np.round((1 / len(self.Xscale)) * (100 * ((cj - ci) / ci)).sum(), 2)
    #
    #     msg_o = 'ΔOutput ' + '  ' + str(me) + '%'
    #
    #     if ii > 0:
    #
    #         plt.figure()
    #         ax = plt.subplot(111)
    #         plt.plot(self.Xscale * 1e3, cj, '-r', linewidth=2.0, label='Altered')
    #         plt.plot(self.Xscale * 1e3, ci, '--b', linewidth=1.0, label='Original')
    #         plt.xlabel('Scaled Distance')
    #         plt.ylabel('Concentration [mM/L]')
    #
    #         # add text to plot describing which parameters were changed in this run,
    #         # and by how much:
    #         if len(delta_i):
    #
    #             for si, (ni, di) in enumerate(zip(name_i, delta_i)):
    #                 msg_i = 'Δ' + ni + '  ' + str(di) + '%'
    #
    #                 plt.text(0.02, 0.95 - (si / 15), msg_i, transform=ax.transAxes)
    #
    #                 #                 Add text describing how output changes for this run:
    #         plt.text(0.02, 0.95 - (len(delta_i) / 15), msg_o,
    #                  transform=ax.transAxes)
    #
    #         plt.savefig(impath_c_init, format='png', dpi=300)
    #         plt.close()
    #
    # # Plot Results
    # def plot_data(self, dataname):
    #
    #     change_input_msgs, change_output_msg, _, _ = self.work_changes(dataname)
    #
    #     for ii in range(self.p.N_runs):
    #
    #         ci = self.outputs[0][dataname]
    #         cj = self.outputs[ii][dataname]
    #
    #         impath = os.path.join(self.savepath, dataname)
    #         os.makedirs(impath, exist_ok=True)
    #
    #         fnme = dataname + str(ii) + '.png'
    #
    #         impath_c_init = os.path.join(impath, fnme)
    #
    #         if ii > 0:
    #
    #             plt.figure()
    #             ax = plt.subplot(111)
    #             plt.plot(self.Xs * 1e3, cj, '-r', linewidth=2.0, label='Altered')
    #             plt.plot(self.Xs * 1e3, ci, '--b', linewidth=1.0, label='Original')
    #             plt.xlabel('Scaled Distance')
    #             plt.ylabel('Concentration [mM/L]')
    #
    #             # add text to plot describing which parameters were changed in this run,
    #             # and by how much:
    #             for si, msgi in enumerate(change_input_msgs[ii]):
    #                 plt.text(0.02, 0.95 - (si / 15), msgi, transform=ax.transAxes)
    #
    #             # Add text describing how output changes for this run:
    #             plt.text(0.02, 0.95 - (len(change_input_msgs[ii]) / 15), change_output_msg[ii],
    #                      transform=ax.transAxes)
    #
    #             plt.savefig(impath_c_init, format='png', dpi=300)
    #             plt.close()
    #
    # def output_delta_table(self, dataname):
    #
    #     _, _, self.change_input, self.change_output = self.work_changes(dataname)
    #
    #     hdr = ''
    #
    #     for plab in self.p.param_labels:
    #         hdr += '% Δ' + plab + ','
    #     hdr += '% ΔOutput'
    #
    #     writeM = np.column_stack((self.change_input[1:], self.change_output[1:]))
    #
    #     fnme = dataname + '_analysis.csv'
    #
    #     fpath = os.path.join(self.savepath, fnme)
    #
    #     np.savetxt(fpath, writeM, delimiter=',', header=hdr)
    #
    # def output_summary_table(self, dataname):
    #
    #     _, _, self.change_input, self.change_output = self.work_changes(dataname)
    #
    #     a, b = self.change_input[1:].shape
    #
    #     if a == b:
    #         diag_vals = np.diag(self.change_input[1:])
    #
    #         hdr = 'Parameter, %Δ Parameter, %Δ Output'
    #
    #         writeM = np.column_stack((self.p.param_labels, diag_vals, self.change_output[1:]))
    #
    #         fnme = dataname + '_sensitiviy_analysis.csv'
    #
    #         fpath = os.path.join(self.savepath, fnme)
    #
    #         np.savetxt(fpath, writeM, fmt='%s', delimiter=',', header=hdr)
    #
    #     head = ''
    #     for pi in self.p.param_labels:
    #         head += pi + ','
    #
    #     fpath2 = os.path.join(self.savepath, 'param_values.csv')
    #     np.savetxt(fpath2, self.p.params_M, delimiter=',', header=head)
    #
    # def work_changes(self, dataname):
    #
    #     msg_inputs = []
    #     msg_outputs = []
    #
    #     tab_inputs = []
    #     tab_outputs = []
    #
    #     for ii in range(self.p.N_runs):
    #
    #         msg_in = []
    #
    #         ci = self.outputs[0][dataname]
    #         cj = self.outputs[ii][dataname]
    #
    #         # calculate percent changes to input variables in this run
    #         percent_delta_input = 100 * ((self.p.params_M[ii, :] -
    #                                       self.p.params_M[0, :]) / self.p.params_M[0, :])
    #
    #         index_delta = (percent_delta_input != 0.0).nonzero()[0]
    #
    #         delta_i = np.round(percent_delta_input[index_delta], 1)
    #         name_i = self.p.param_labels[index_delta]
    #
    #         for si, (ni, di) in enumerate(zip(name_i, delta_i)):
    #             msg = 'Δ' + ni + '  ' + str(di) + '%'
    #
    #             msg_in.append(msg)
    #
    #         me = np.round((1 / len(self.X)) * (100 * ((cj - ci) / ci)).sum(), 2)
    #
    #         msg_o = 'ΔOutput ' + '  ' + str(me) + '%'
    #
    #         msg_outputs.append(msg_o)
    #
    #         msg_inputs.append(msg_in)
    #
    #         tab_outputs.append(me * 1)
    #         tab_inputs.append(percent_delta_input * 1)
    #
    #     tab_inputs = np.asarray(tab_inputs)
    #     tab_outputs = np.asarray(tab_outputs)
    #
    #     return msg_inputs, msg_outputs, tab_inputs, tab_outputs
    #
    # def render_output(self, makeplots=True):
    #
    #     for tag in self.datatags:  # Save plots of each concentration to disk
    #
    #         if makeplots:
    #             self.plot_data(tag)
    #
    #         # Save spreadsheet of model changes for parameter changes:
    #         self.output_delta_table(tag)
    #
    #         self.output_summary_table(tag)


