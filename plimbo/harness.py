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

    def __init__(self, config_filename, paramo = None, xscale=1.0, harness_type = '1D', plot_frags = True,
                 verbose = False, new_mesh=False, savedir = 'ModelSearch', head_frags = None, tail_frags = None):

        self.xscale = xscale

        self.paramo = paramo

        self.verbose = verbose

        self.new_mesh = new_mesh

        self.savedir = savedir

        self.config_fn = config_filename

        # specify fragments that are heads or tails for the Markov simulation:
        if head_frags is None:
            self.head_frags = [0]

        else:
            self.head_frags = head_frags

        if tail_frags is None:
            self.tail_frags = [4]

        else:
            self.tail_frags = tail_frags

        # Create a simulator object:
        if harness_type == '1D':
            self.model = PlanariaGRN1D(config_filename, self.paramo, xscale=1.0, verbose=False, new_mesh=False)

        if harness_type == '2D':
            self.model = PlanariaGRN2D(config_filename, self.paramo, xscale=1.0, verbose=False, new_mesh=False)

        if self.paramo is None:
            self.paramo = self.model.pdict

        # save the harness type so we know what we're working with:
        self.harness_type = harness_type

        # Create a directory to store results:
        self.savepath = pathnames.join_and_canonicalize(self.model.p.conf_dirname, savedir)

        os.makedirs(self.savepath, exist_ok=True)

        # Create a dictionary to save sub-folder paths:
        self.subfolders_dict = OrderedDict()

        # Generate a parameters manager object for the harness:
        self.pm = ParamsManager(self.paramo)

        self.has_autoparams = False

        # initialize outputs array to null:
        self.outputs = []

        # save the model's concentration tags:
        self.conctags = self.model.conc_tags

        # reference data set to null
        self.ref_data = None

        # extra information to write on plots:
        self.plot_info_msg = None


        # default RNAi testing sequence vector:
        self.RNAi_vect_default = [
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 0.25,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 5.0,
             'dynein': 1, 'kinesin': 1},
            {'bc': 0.1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 0.01, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 0.01, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 0.01, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 0.01, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 0.01, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 0.01, 'camp': 1,
             'dynein': 1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 0.1, 'kinesin': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 0.1},
        ]

        self.RNAi_tags_default = ['cAMP_0.25x', 'cAMP_5x', 'RNAi_BC_part',  'RNAi_BC', 'RNAi_ERK', 'RNAi_APC',
                                  'RNAi_Notum', 'RNAi_WNT', 'RNAi_HH', 'Dynein', 'Kinesin']

        self.xscales_default = [0.75, 1.5, 3.0]



    def run_sensitivity(self, factor = 0.1, verbose=True, run_time_init = 36000.0,
                        run_time_sim = 36000.0, run_time_step = 60, run_time_sample = 50,
                        reset_clims = True, animate = False, plot = True, plot_type = 'Triplot',
                        save_dir = 'Sensitivity1', ani_type = 'Triplot', save_all = False, data_output = True):

        # general saving directory for this procedure:
        self.savedir_sensitivity = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_sensitivity, exist_ok=True)

        self.subfolders_dict['sensitivity'] = self.savedir_sensitivity

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        self.pm.create_sensitivity_matrix(factor=factor)  # Generate sensitivity matrix from default parameters
        self.has_autoparams = True

        self.outputs = []  # Storage array for all data created in each itteration of the model
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            try:

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

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

                # if we're on the first timestep, set it as the reference data set:
                if ii == 0:
                    self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                # if we're past the first timesep, prepare messages for the plots about how params have changed:
                if ii > 0:
                    self.write_plot_msg(ii)

                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                self.outputs.append([data_dict_inits, data_dict_sims])

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                self.heteromorphoses.append(data_dict_prob)


                if plot:

                    self.plot_single('base', ii, harness_type='sensitivity', plot_type=plot_type,
                                     output_type='init', ref_data=self.ref_data[0], extra_text = self.plot_info_msg)

                    self.plot_single('base', ii, harness_type='sensitivity', plot_type=plot_type,
                                     output_type='sim', ref_data=self.ref_data[1], extra_text = self.plot_info_msg)

                if animate:
                    self.ani_single('base', ii, harness_type='sensitivity', ani_type=ani_type,
                                    output_type='init', ref_data=self.ref_data[0], extra_text = self.plot_info_msg)
                    self.ani_single('base', ii, harness_type='sensitivity', ani_type=ani_type,
                                    output_type='sim', ref_data=self.ref_data[1], extra_text = self.plot_info_msg)

                if verbose is True:
                    print('----------------')


            except:
                print('***************************************************')
                print("Run", ii + 1, "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_delta_table(substance='Head', run_type='init', save_dir=self.savedir_sensitivity)
            self.output_summary_table(substance='Head', run_type='init', save_dir=self.savedir_sensitivity)
            self.output_heteromorphs(save_dir=self.savedir_sensitivity)

        if save_all:
            fsave = os.path.join(self.savedir_sensitivity, "Master.gz")
            self.save(fsave)


    def run_search(self, factor = 0.8, levels = 1, search_style = 'log', verbose=True,
                   run_time_init=36000.0, run_time_sim=36000.0, run_time_step=60, run_time_sample=50,
                   reset_clims=True, plot=True, animate = False, save_dir = 'Search1', save_all = False,
                   fixed_params = None, plot_type = 'Triplot', ani_type = 'Triplot', data_output = True):

        # general saving directory for this procedure:
        self.savedir_search = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_search, exist_ok=True)

        self.subfolders_dict['search'] = self.savedir_search

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        # Create the parameters matrix:
        self.pm.create_search_matrix(factor=factor, levels=levels, style=search_style)
        self.has_autoparams = True

        self.outputs = []  # Storage array for all last timestep outputs
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", self.pm.N_runs)

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

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

                # if we're on the first timestep, set the reference data set:
                if ii == 0:
                    self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                # if we're past the first timesep, prepare messages for the plots about how params have changed:
                if ii > 0:
                    self.write_plot_msg(ii)


                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                self.outputs.append([data_dict_inits, data_dict_sims])

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()
                self.heteromorphoses.append(data_dict_prob)

                if plot:
                    self.plot_single('base', ii, harness_type='search', plot_type=plot_type, output_type='init',
                                     ref_data = self.ref_data[0], extra_text = self.plot_info_msg)
                    self.plot_single('base', ii, harness_type='search', plot_type=plot_type, output_type='sim',
                                     ref_data=self.ref_data[1], extra_text = self.plot_info_msg)

                if animate:
                    self.ani_single('base', ii, harness_type='search', ani_type=ani_type,
                                    output_type='init', ref_data = self.ref_data[0], extra_text = self.plot_info_msg)
                    self.ani_single('base', ii, harness_type='search', ani_type=ani_type,
                                    output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg)

                if verbose is True:
                    print('----------------')

            except:

                print('***************************************************')
                print("Run", ii +1, "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_delta_table(substance='Head', run_type='init', save_dir=self.savedir_search)
            self.output_summary_table(substance='Head', run_type='init', save_dir=self.savedir_search)
            self.output_heteromorphs(save_dir=self.savedir_sensitivity)

        if save_all:
            fsave = os.path.join(self.savedir_search, "Master.plimbo")
            self.save(fsave)

    def run_searchRNAi(self, RNAi_series = None, RNAi_names = None, factor = 0.8, levels = 1, search_style = 'log',
                        verbose=True, run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0, save_all = False,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True, ani_type = 'Triplot',
                       animate=False, save_dir='SearchRNAi1', fixed_params = None, plot_type = 'Triplot',
                       data_output = True):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

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
        self.has_autoparams = True

        self.outputs = []  # Storage array for all last timestep outputs
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", self.pm.N_runs)

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

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

                # if we're on the first timestep, set it as the reference data set:
                if ii == 0:
                    self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                # if we're past the first timesep, prepare messages for the plots about how params have changed:
                if ii > 0:
                    self.write_plot_msg(ii)


                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                if plot:
                    self.plot_single('base', ii, harness_type='searchRNAi', plot_type=plot_type,
                                     output_type='init', ref_data = self.ref_data[0], extra_text = self.plot_info_msg)
                    self.plot_single('base', ii, harness_type='searchRNAi', plot_type=plot_type,
                                     output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg)

                if animate:
                    self.ani_single('base', ii, harness_type='searchRNAi', ani_type=ani_type,
                                    output_type='init', ref_data = self.ref_data[0], extra_text = self.plot_info_msg)
                    self.ani_single('base', ii, harness_type='searchRNAi', ani_type=ani_type,
                                    output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg)

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

                    self.model.process_markov(self.head_frags, self.tail_frags)
                    data_dict_prob[rnai_n] = self.model.morph_probs.copy()

                    if plot:
                        self.plot_single(rnai_n, ii, harness_type='searchRNAi', plot_type=plot_type,
                                         output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg)

                    if animate:
                        self.ani_single(rnai_n, ii, harness_type='searchRNAi', ani_type=ani_type,
                                        output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg)

                    if verbose is True:
                        print('----------------')

                self.outputs.append([data_dict_inits, data_dict_sims])
                self.heteromorphoses.append(data_dict_prob)

            except:

                print('***************************************************')
                print("Run", ii +1, "has become unstable and been terminated.")
                print('***************************************************')


        if data_output:
            self.output_delta_table(substance='Head', run_type='init', save_dir=self.savedir_searchRNAi)
            self.output_summary_table(substance='Head', run_type='init', save_dir=self.savedir_searchRNAi)
            self.output_heteromorphs(save_dir=self.savedir_sensitivity)

        if save_all:
            fsave = os.path.join(self.savedir_searchRNAi, "Master.gz")
            self.save(fsave)

    def run_scale(self, xscales = None, verbose=True,
                       run_time_init=36000.0, run_time_sim=36000.0,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True,
                       animate=False, save_dir='scale1', plot_type = 'Triplot',
                       ani_type = 'Triplot', save_all = False, data_output = True
                       ):

        if xscales is None:
            xscales = self.xscales_default

        # general saving directory for this procedure:
        self.savedir_scale = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_scale, exist_ok=True)

        self.subfolders_dict['scale'] = self.savedir_scale

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        self.outputs = []  # Storage array for all last timestep outputs
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, xxs in enumerate(xscales):  # Step through the x-scaling factors

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", len(xscales))

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

                # create a model using the specific parameters from the params manager for this run at this scale:
                self.model.model_init(self.config_fn, self.paramo, xscale=xxs,
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

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()
                self.heteromorphoses.append(data_dict_prob)

                if plot:
                    self.plot_single('base', ii, harness_type='scale', plot_type=plot_type,
                                     output_type='init', extra_text = self.plot_info_msg)
                    self.plot_single('base', ii, harness_type='scale', plot_type=plot_type,
                                     output_type='sim', extra_text = self.plot_info_msg)

                if animate:
                    self.ani_single('base', ii, harness_type='scale', ani_type=ani_type,
                                    output_type='init', extra_text = self.plot_info_msg)
                    self.ani_single('base', ii, harness_type='scale', ani_type=ani_type,
                                    output_type='sim', extra_text = self.plot_info_msg)

                if verbose is True:
                    print('----------------')

            except:

                print('***************************************************')
                print("Run", ii +1, "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_heteromorphs(save_dir=self.savedir_scale)

        if save_all:
            fsave = os.path.join(self.savedir_scale, "Master.gz")
            self.save(fsave)

    def run_scaleRNAi(self, xscales = None, RNAi_series = None, RNAi_names = None, verbose=True,
                       run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True,
                       animate=False, save_dir='scaleRNAi1', plot_type = 'Triplot', save_all = False,
                       ani_type = 'Triplot', data_output = True
                       ):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

        if xscales is None:
            xscales = self.xscales_default

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
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, xxs in enumerate(xscales):  # Step through the x-scaling factors

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", len(xscales))

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

                # create a model using the specific parameters from the params manager for this run at this scale:
                self.model.model_init(self.config_fn, self.paramo, xscale=xxs,
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

                # Set reference data set to the main model curves::
                self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                if plot:
                    self.plot_single('base', ii, harness_type='scaleRNAi', plot_type=plot_type,
                                     output_type='init', extra_text = self.plot_info_msg)
                    self.plot_single('base', ii, harness_type='scaleRNAi', plot_type=plot_type,
                                     output_type='sim', extra_text = self.plot_info_msg)

                if animate:
                    self.ani_single('base', ii, harness_type='scaleRNAi', ani_type=ani_type,
                                    output_type='init', extra_text = self.plot_info_msg)
                    self.ani_single('base', ii, harness_type='scaleRNAi', ani_type=ani_type,
                                    output_type='sim', extra_text = self.plot_info_msg)

                if verbose is True:
                    print('----------------')

                for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                    if verbose is True:
                        print('Runing RNAi Sequence ', rnai_n)

                    # Reinitialize the model again:
                    self.model.model_init(self.config_fn, self.paramo, xscale=xxs,
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

                    self.model.process_markov(self.head_frags, self.tail_frags)
                    data_dict_prob[rnai_n] = self.model.morph_probs.copy()

                    if plot:
                        self.plot_single(rnai_n, ii, harness_type='scaleRNAi', plot_type=plot_type,
                                         output_type='sim', ref_data=self.ref_data[1])

                    if animate:
                        self.ani_single(rnai_n, ii, harness_type='scaleRNAi', ani_type=ani_type,
                                        output_type='sim', ref_data=self.ref_data[1])

                    if verbose is True:
                        print('----------------')

                self.outputs.append([data_dict_inits, data_dict_sims])
                self.heteromorphoses.append(data_dict_prob)

            except:

                print('***************************************************')
                print("Run", ii +1 , "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_heteromorphs(save_dir=self.savedir_scaleRNAi)


        if save_all:
            fsave = os.path.join(self.savedir_scaleRNAi, "Master.gz")
            self.save(fsave)

    def run_simRNAi(self, RNAi_series = None, RNAi_names = None, verbose=True,
                    run_time_init = 36000.0, run_time_sim = 36000.0, run_time_step = 60,
                    run_time_sample = 50, run_time_reinit = 12, reset_clims = True,
                    plot_type = 'Triplot', ani_type = 'Triplot', animate = False, save_all = False,
                    plot = True, save_dir = 'SimRNAi_1', data_output = True):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

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
        self.heteromorphoses = [] # Storage array for heteromorph probabilities
        self.wound_probs = []

        data_dict_inits = OrderedDict() # storage of inits for each molecules of a model itterantion
        data_dict_sims = OrderedDict() # storage of sims for each molecules of a model itterantion
        data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

        # create a model using the specific parameters from the params manager for this run:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
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

        # Set reference data set to the main model curves::
        self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

        self.model.process_markov(self.head_frags, self.tail_frags)
        data_dict_prob['base'] = self.model.morph_probs.copy()

        self.wound_probs.append(self.model.frag_probs.copy())

        if plot:
            self.plot_single('base', 0, harness_type='simRNAi', plot_type=plot_type,
                             output_type='init', extra_text = self.plot_info_msg)
            self.plot_single('base', 0, harness_type='simRNAi', plot_type=plot_type,
                             output_type='sim', extra_text = self.plot_info_msg)

        if animate:
            self.ani_single('base', 0, harness_type='simRNAi', ani_type=ani_type,
                            output_type='init', extra_text = self.plot_info_msg)
            self.ani_single('base', 0, harness_type='simRNAi', ani_type=ani_type,
                            output_type='sim', extra_text = self.plot_info_msg)

        if verbose is True:
            print('----------------')

        for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

            if verbose is True:
                print('Runing RNAi Sequence ', rnai_n)

            # Reinitialize the model again:
            self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
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

            self.model.process_markov(self.head_frags, self.tail_frags)
            data_dict_prob[rnai_n] = self.model.morph_probs.copy()

            if plot:
                self.plot_single(rnai_n, 0, harness_type='simRNAi', plot_type=plot_type,
                                 output_type='sim', ref_data=self.ref_data[1])


            if animate:
                self.ani_single(rnai_n, 0, harness_type='simRNAi', ani_type=ani_type,
                                output_type='sim', ref_data=self.ref_data[1])

            if verbose is True:
                print('----------------')


        self.outputs.append([data_dict_inits, data_dict_sims])
        self.heteromorphoses.append(data_dict_prob)

        if data_output:
            self.output_heteromorphs(save_dir=self.savedir_simRNAi)

        if save_all:

            fsave = os.path.join(self.savedir_simRNAi, "Master.gz")
            self.save(fsave)


    def save(self, fname):

        if self.verbose:
            print("Saving harness results to file...")

        pickles.save(self, filename=fname, is_overwritable=True)

    def load(self, fname):

        if self.verbose:
            print("Loaded saved harness...")

        master = pickles.load(fname)

        return master

    def plot_all_output(self, loadpath, save_dir = 'Plots', plot_type='Triplot', output_type='sim',
                        autoscale = False, clims = None, cmaps = None):

        load_fname = os.path.join(loadpath, "Master.gz")
        master = self.load(load_fname)

        dirname = os.path.join(loadpath, save_dir)

        if output_type == 'init':
            ref_data = master.ref_data[0]
        elif output_type == 'sim':
            ref_data = master.ref_data[1]

        if len(master.outputs):

            if output_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for init_i, tagi in zip(inits_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Plotting init of', tagi, 'for run ', ri, '...')

                        fni = plot_type + '_' + tagi + '_init_' + str(ri)

                        master.model.molecules_time = init_i

                        if ri > 0:
                            master.write_plot_msg(ri)

                        if plot_type == 'Triplot':
                            master.model.triplot(-1, plot_type='init', fname=fni, dirsave=dirname,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale,
                                               ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif plot_type == 'Biplot':
                            master.model.biplot(-1, plot_type='init', fname=fni, dirsave=dirname,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale,
                                              ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif plot_type == 'Markovplot':
                            master.model.markovplot(-1, plot_type='init', fname=fni, dirsave=dirname,
                                                  cmaps=None, clims=None, autoscale=False,
                                                  ref_data=ref_data, extra_text=master.plot_info_msg)

            elif output_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    if ri > 0:
                        master.write_plot_msg(ri)

                    for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Plotting sim of', tagi, 'for run ', ri, '...')

                        fns = plot_type + '_' + tagi + '_sim_' + str(ri)
                        master.model.molecules_sim_time = sim_i

                        if plot_type == 'Triplot':
                            master.model.triplot(-1, plot_type='sim', fname=fns, dirsave=dirname,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale,
                                               ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif plot_type == 'Biplot':
                            master.model.biplot(-1, plot_type='sim', fname=fns, dirsave=dirname,
                                              cmaps=cmaps, clims=clims, autoscale=autoscale,
                                              ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif plot_type == 'Markovplot':
                            master.model.markovplot(-1, plot_type='sim', fname=fns, dirsave=dirname,
                                                  cmaps=None, clims=None, autoscale=False,
                                                  ref_data=ref_data, extra_text=master.plot_info_msg)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to plot.')

    def ani_all_output(self, loadpath, save_dir = 'Animations', ani_type='Triplot', output_type='sim',
                       autoscale = False, cmaps = None, clims = None):

        load_fname = os.path.join(loadpath, "Master.gz")
        master = self.load(load_fname)

        dirname = os.path.join(loadpath, save_dir)

        if output_type == 'init':
            ref_data = master.ref_data[0]
        elif output_type == 'sim':
            ref_data = master.ref_data[1]


        if len(master.outputs):

            if output_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for init_i, tagi in zip(inits_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Animating init of', tagi, 'for run ', ri, '...')

                        dni = ani_type + '_' + tagi + '_init'+ str(ri)
                        plotdiri = os.path.join(dirname, dni)

                        master.model.molecules_time = init_i

                        if ri > 0:
                            master.write_plot_msg(ri)

                        if ani_type == 'Triplot':
                            self.model.animate_triplot(ani_type='init', dirsave=plotdiri,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale,
                                                       ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif ani_type == 'Biplot':
                            self.model.animate_biplot(ani_type='init', dirsave=plotdiri,
                                              cmaps=cmaps, clims=clims, autoscale=autoscale,
                                                      ref_data = ref_data, extra_text = master.plot_info_msg)

            elif output_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Plotting sim of', tagi, 'for run ', ri, '...')

                        dns = ani_type + '_' + tagi + '_sim' + str(ri)
                        plotdirs =os.path.join(dirname, dns)

                        master.model.molecules_sim_time = sim_i

                        if ri > 0:
                            master.write_plot_msg(ri)

                        if ani_type == 'Triplot':
                            master.model.animate_triplot(ani_type='sim', dirsave=plotdirs,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale,
                                                       ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif ani_type == 'Biplot':
                            master.model.animate_biplot(ani_type='sim', dirsave=plotdirs,
                                              cmaps=cmaps, clims=clims, autoscale=autoscale,
                                                      ref_data = ref_data, extra_text = master.plot_info_msg)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to animate.')

    def plot_single(self, tagi, ri, harness_type=None, plot_type='Triplot', output_type='sim',
                    ref_data = None, extra_text = None):
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
                                   cmaps=None, clims=None, autoscale=False,
                                   ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Biplot':
                self.model.biplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False,
                                  ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Hexplot':
                self.model.hexplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False,
                                  ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Markovplot':
                self.model.markovplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False,
                                   ref_data = ref_data, extra_text = extra_text)

        elif output_type == 'sim':

            if self.verbose is True:
                print('Plotting sim of', tagi, 'for run ', ri, '...')

            fns = plot_type + '_' + tagi + '_sim_' + str(ri)

            if plot_type == 'Triplot':
                self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False,
                                   ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Biplot':
                self.model.biplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                  cmaps=None, clims=None, autoscale=False,
                                  ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Hexplot':
                self.model.hexplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                  cmaps=None, clims=None, autoscale=False,
                                  ref_data = ref_data, extra_text = extra_text)

            elif plot_type == 'Markovplot':
                self.model.markovplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain,
                                   cmaps=None, clims=None, autoscale=False,
                                   ref_data = ref_data, extra_text = extra_text)

    def ani_single(self, tagi, ri, harness_type=None, ani_type='Triplot', output_type='sim',
                   ref_data = None, extra_text = None):

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
                                   cmaps=None, clims=None, autoscale=False,
                                           ref_data = ref_data, extra_text = extra_text)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='init', dirsave=plotdiri,
                                  cmaps=None, clims=None, autoscale=False,
                                          ref_data = ref_data, extra_text = extra_text)

        elif output_type == 'sim':

            if self.verbose is True:
                print('Animating sim of', tagi, 'for run ', ri, '...')

            dns = ani_type + '_' + tagi + '_sim'+ str(ri)
            plotdirs =os.path.join(plotdirmain, dns)

            if ani_type == 'Triplot':
                self.model.animate_triplot(ani_type='sim', dirsave=plotdirs,
                                   cmaps=None, clims=None, autoscale=False,
                                           ref_data = ref_data, extra_text = extra_text)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='sim', dirsave=plotdirs,
                                  cmaps=None, clims=None, autoscale=False,
                                          ref_data = ref_data, extra_text = extra_text)

    def write_plot_msg(self, ii):

        if self.has_autoparams:

            # calculate percent changes to input variables in this run
            percent_delta_input = 100 * ((self.pm.params_M[ii, :] -
                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

            index_delta = (percent_delta_input != 0.0).nonzero()[0]

            delta_i = np.round(percent_delta_input[index_delta], 1)
            name_i = self.pm.param_labels[index_delta]

            # add text to plot describing which parameters were changed in this run,
            # and by how much:
            param_changes_msg = ''
            if len(delta_i):

                for si, (ni, di) in enumerate(zip(name_i, delta_i)):
                    msg_i = 'Δ' + ni + '  ' + str(di) + '%\n'
                    param_changes_msg += msg_i

            self.plot_info_msg = param_changes_msg


    def work_all_output(self, save_dir = 'DataOutput', ii =-1, run_type = 'init',
                        substance = 'Erk', ref_data = None):

        tab_inputs = []
        tab_outputs = []

        master = self

        if len(master.outputs):

            if run_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for init_i, tagi in zip(inits_dict.values(), master.datatags):

                        master.model.molecules_time = init_i

                        # if ri > 0:
                        out_diff = master.model.work_data(ti =ii, run_type = 'init',
                                                          substance = substance, ref_data = ref_data)

                        tab_outputs.append(out_diff * 1)

                        if self.has_autoparams:
                            # calculate percent changes to input variables in this run:
                            percent_delta_input = 100 * ((self.pm.params_M[ri, :] -
                                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

                        else:
                            percent_delta_input = 0.0

                        tab_inputs.append(percent_delta_input * 1)

            elif run_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                        master.model.molecules_sim_time = sim_i

                        # if ri > 0:
                        out_diff = master.model.work_data(ti=ii, run_type='sim',
                                                          substance=substance, ref_data=ref_data)

                        tab_outputs.append(out_diff * 1)

                        if self.has_autoparams:
                            # calculate percent changes to input variables in this run
                            percent_delta_input = 100 * ((self.pm.params_M[ri, :] -
                                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

                        else:
                            percent_delta_input = 0.0

                        tab_inputs.append(percent_delta_input * 1)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to work.')

        tab_inputs = np.asarray(tab_inputs)
        tab_outputs = np.asarray(tab_outputs)

        return tab_inputs, tab_outputs


    def output_delta_table(self, substance = 'Erk', run_type = 'sim', save_dir = 'DataOutput'):

        change_input, change_output = self.work_all_output(substance = substance,
                                                                     run_type = run_type,
                                                                     ref_data = self.ref_data)

        hdr = ''

        for plab in self.pm.param_labels:
            hdr += '% Δ' + plab + ','
        hdr += '% ΔOutput'

        writeM = np.column_stack((change_input[1:], change_output[1:]))

        fnme = substance + '_analysis.csv'

        fpath = os.path.join(save_dir, fnme)

        np.savetxt(fpath, writeM, delimiter=',', header=hdr)

    def output_summary_table(self, substance = 'Erk', run_type = 'sim', save_dir = 'DataOutput'):

        change_input, change_output = self.work_all_output(substance=substance,
                                                                     run_type = run_type,
                                                                     ref_data = self.ref_data)

        a, b = change_input[1:].shape

        if a == b:
            diag_vals = np.diag(change_input[1:])

            hdr = 'Parameter, %Δ Parameter, %Δ Output'

            writeM = np.column_stack((self.pm.param_labels, diag_vals, change_output[1:]))

            fnme = substance + '_sensitiviy_analysis.csv'

            fpath = os.path.join(save_dir, fnme)

            np.savetxt(fpath, writeM, fmt='%s', delimiter=',', header=hdr)

        head = ''
        for pi in self.pm.param_labels:
            head += pi + ','

        fpath2 = os.path.join(save_dir, 'param_values.csv')
        np.savetxt(fpath2, self.pm.params_M, delimiter=',', header=head)


    def output_heteromorphs(self, save_dir = 'DataOutput'):

        morph_col_tags = '2T,' + '0H,' + '1H,' + '0T,' + '2H'

        for ri, hetmorphs_masterdict in enumerate(self.heteromorphoses):

            dir_path = os.path.join(save_dir, 'Run_' + str(ri))
            os.makedirs(dir_path, exist_ok=True)

            for tag_n, hmorphs_dict in hetmorphs_masterdict.items():

                morph_data = []

                for frag_n, hmorphs, in hmorphs_dict.items():
                    p2T = np.round(hmorphs['2T'], 2)
                    p0H = np.round(hmorphs['0H'], 2)
                    p1H = np.round(hmorphs['1H'], 2)
                    p0T = np.round(hmorphs['0T'], 2)
                    p2H = np.round(hmorphs['2H'], 2)

                    row_data = [p2T, p0H, p1H, p0T, p2H]

                    morph_data.append(row_data)

                morph_data = np.asarray(morph_data)

                fpath = os.path.join(dir_path, tag_n + '_heteromorphs.csv')

                np.savetxt(fpath, morph_data, delimiter=',', header=morph_col_tags)

    def view_fragments(self):

        # create a model using the specific parameters from the params manager for this run:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                              verbose=self.verbose, new_mesh=self.new_mesh)

        if self.harness_type == '2D':

            self.model.cut_cells()

            self.model.plot_frags(dir_save=self.savepath)




