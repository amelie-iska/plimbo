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

        # Create a directory to store results:
        self.savepath = pathnames.join_and_canonicalize(self.model.p.conf_dirname, savedir)

        os.makedirs(self.savepath, exist_ok=True)

        # Generate a parameters manager object for the harness:
        self.pm = ParamsManager(self.paramo)


    def run_sensitivity_analysis(self, factor = 0.1, verbose=True,
                                 run_time_init = 36000.0, run_time_step = 60, run_time_sample = 50,
                                 reset_clims = True, animate = False, plot = True, save_dir = 'Sensitivity1'):

        # general saving directory for this procedure:
        self.savedir_sensitivity = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_sensitivity, exist_ok=True)


        if plot:
            savedir_plot = os.path.join(self.savedir_sensitivity, 'Triplots')
            os.makedirs(savedir_plot, exist_ok=True)

        # Create datatags for the harness to save data series to:
        self.datatags = []

        for tag in self.model.conc_tags:
            data_tag = tag + '_init'
            self.datatags.append(data_tag)

        self.pm.create_sensitivity_matrix(factor = factor)  # Generate sensitivity matrix from default parameters

        self.outputs = []  # Storage array for all last timestep outputs

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

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

            data_dict = OrderedDict()

            for tag in self.model.conc_tags:
                data_tag = tag + '_init'
                data_dict[data_tag] = self.model.molecules_time[tag][-1]

            self.outputs.append(data_dict)

            self.X = self.model.X
            self.Xscale = self.model.X / self.model.xmax
            self.seg_inds = self.model.seg_inds

            if plot:

                fn = 'Triplot_run' + str(ii) + '_t'

                self.model.triplot(-1, plot_type='init', fname = fn, dirsave=savedir_plot,
                        cmaps=None, clims=None, autoscale=True)

            if animate:
                dn = 'TriplotAni_run' + str(ii)

                savedir_ani = os.path.join(self.savedir_sensitivity, dn)
                os.makedirs(savedir_ani, exist_ok=True)

                self.model.animate_triplot(ani_type='init', dirsave=savedir_ani,
                                cmaps=None, clims=None, autoscale=True)


    # def run_search_analysis(self, verbose=True, plotwhilesolving=True):
    #
    #     # Datatags run by this harness:
    #     # Create datatags for the harness to save data series to:
    #     self.datatags = []
    #
    #     for tag in self.model.conc_tags:
    #         data_tagi = tag + '_init'
    #         data_tags = tag + '_sim'
    #         self.datatags.append(data_tagi)
    #         self.datatags.append(data_tags)
    #
    #     # Create the parameters matrix:
    #     self.pm.create_search_matrix(style=self.search_style)
    #
    #     self.outputs = []  # Storage array for all last timestep outputs
    #
    #     for ii, params_list in enumerate(self.p.params_M):  # Step through the parameters matrix
    #
    #         if verbose is True:
    #             print('Run ', ii + 1, " of ", self.p.N_runs)
    #
    #         # convert the array to a dictionary:
    #         run_params = OrderedDict(zip(self.p.param_labels, params_list))
    #
    #         model = PlanariaGRN1D(run_params, xscale=self.xscale)  # create a model using the specific parameters
    #
    #         # Run initialization of full model:
    #         model.initialize(knock_down=self.RNAi,
    #                          run_time=self.runtime,
    #                          run_time_step=self.runstep,
    #                          run_time_sample=self.runsample)
    #
    #         model.simulate(knock_down=self.RNAi,
    #                        run_time=self.runtime,
    #                        run_time_step=self.runstep,
    #                        run_time_sample=self.runsample)
    #
    #         data_dict = OrderedDict()
    #
    #         # FIXME save whole time series from each one!
    #         data_dict['bc_init'] = model.c_BC_time[-1]
    #         data_dict['wnt_init'] = model.c_WNT_time[-1]
    #         data_dict['notum_init'] = model.c_Notum_time[-1]
    #         data_dict['nrf_init'] = model.c_NRF_time[-1]
    #         data_dict['hh_init'] = model.c_HH_time[-1]
    #         data_dict['erk_init'] = model.c_ERK_time[-1]
    #         data_dict['apc_init'] = model.c_APC_time[-1]
    #         data_dict['camp_init'] = model.c_cAMP_time[-1]
    #
    #         data_dict['bc_sim'] = model.c_BC_sim_time[-1]
    #         data_dict['wnt_sim'] = model.c_WNT_sim_time[-1]
    #         data_dict['notum_sim'] = model.c_Notum_sim_time[-1]
    #         data_dict['nrf_sim'] = model.c_NRF_sim_time[-1]
    #         data_dict['hh_sim'] = model.c_HH_sim_time[-1]
    #         data_dict['erk_sim'] = model.c_ERK_sim_time[-1]
    #         data_dict['apc_sim'] = model.c_APC_sim_time[-1]
    #         data_dict['camp_sim'] = model.c_cAMP_sim_time[-1]
    #
    #         self.X = model.X
    #         self.Xscale = model.X / model.xmax
    #
    #         self.outputs.append(data_dict)
    #
    #         if plotwhilesolving:
    #
    #             for tag in self.datatags:
    #                 self.plot_indy(ii, tag)
    #
    # def search_and_rnai(self, RNAi_series, RNAi_names,
    #                     verbose=True, plotwhilesolving=True, t_reinit=0.0):
    #
    #     # Create datatags for the harness to save data series to:
    #     self.datatags = []
    #
    #     for tag in self.model.conc_tags:
    #         data_tagi = tag + '_init'
    #         data_tags = tag + '_sim'
    #         self.datatags.append(data_tagi)
    #         self.datatags.append(data_tags)
    #
    #     # add in new datatags for RNAi_series
    #     for rnai_n in RNAi_names:
    #         self.datatags.append(rnai_n)
    #
    #     # Create the parameters matrix:
    #     self.pm.create_search_matrix(style=self.search_style)
    #
    #     self.outputs = []  # Storage array for all last timestep outputs
    #
    #     for ii, params_list in enumerate(self.p.params_M):  # Step through the parameters matrix
    #
    #         if verbose is True:
    #             print('Run ', ii + 1, " of ", self.p.N_runs)
    #
    #         # convert the array to a dictionary:
    #         run_params = OrderedDict(zip(self.p.param_labels, params_list))
    #
    #         model = PlanariaGRN1D(run_params, xscale=self.xscale)  # create a model using the specific parameters
    #
    #         # Run initialization of full model:
    #         model.initialize(knock_down=self.RNAi,
    #                          run_time=self.runtime,
    #                          run_time_step=self.runstep,
    #                          run_time_sample=self.runsample)
    #
    #         model.simulate(knock_down=self.RNAi,
    #                        run_time=self.runtime,
    #                        run_time_step=self.runstep,
    #                        run_time_sample=self.runsample)
    #
    #         data_dict = OrderedDict()
    #
    #         data_dict['bc_init'] = model.c_BC_time[-1]
    #         data_dict['wnt_init'] = model.c_WNT_time[-1]
    #         data_dict['notum_init'] = model.c_Notum_time[-1]
    #         data_dict['nrf_init'] = model.c_NRF_time[-1]
    #         data_dict['hh_init'] = model.c_HH_time[-1]
    #         data_dict['erk_init'] = model.c_ERK_time[-1]
    #         data_dict['apc_init'] = model.c_APC_time[-1]
    #         data_dict['camp_init'] = model.c_cAMP_time[-1]
    #
    #         data_dict['bc_sim'] = model.c_BC_sim_time[-1]
    #         data_dict['wnt_sim'] = model.c_WNT_sim_time[-1]
    #         data_dict['notum_sim'] = model.c_Notum_sim_time[-1]
    #         data_dict['nrf_sim'] = model.c_NRF_sim_time[-1]
    #         data_dict['hh_sim'] = model.c_HH_sim_time[-1]
    #         data_dict['erk_sim'] = model.c_ERK_sim_time[-1]
    #         data_dict['apc_sim'] = model.c_APC_sim_time[-1]
    #         data_dict['camp_sim'] = model.c_cAMP_sim_time[-1]
    #
    #         self.X = model.X
    #         self.Xscale = model.X / model.xmax
    #
    #         for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):
    #
    #             if verbose is True:
    #                 print('Runing RNAi Sequence ', rnai_n)
    #
    #             model2 = PlanariaGRN(run_params, xscale=self.xscale)  # create a model using the specific parameters
    #
    #             # Run a new initialization of full model:
    #             model2.initialize(knock_down=self.RNAi,
    #                               run_time=self.runtime,
    #                               run_time_step=self.runstep,
    #                               run_time_sample=self.runsample)
    #
    #             if t_reinit > 0.0:
    #                 model2.reinitialize(knock_down=rnai_s,
    #                                     run_time=t_reinit,
    #                                     run_time_step=self.runstep,
    #                                     run_time_sample=self.runsample)
    #
    #             model2.simulate(knock_down=rnai_s,
    #                             run_time=self.runtime,
    #                             run_time_step=self.runstep,
    #                             run_time_sample=self.runsample)
    #
    #             # FIXME later we want this to be Markov Model output
    #             data_dict[rnai_n] = model2.c_ERK_sim_time[-1] * 1
    #
    #         self.outputs.append(data_dict)
    #
    #         if plotwhilesolving:
    #
    #             for tag in self.datatags:
    #                 self.plot_indy(ii, tag)
    #
    # def scale_and_rnai(self, run_params, xscales, RNAi_series, RNAi_names,
    #                    verbose=True, plotwhilesolving=True, t_reinit=0.0):
    #
    #     # Create datatags for the harness to save data series to:
    #     self.datatags = []
    #
    #     for tag in self.model.conc_tags:
    #         data_tagi = tag + '_init'
    #         data_tags = tag + '_sim'
    #         self.datatags.append(data_tagi)
    #         self.datatags.append(data_tags)
    #
    #     # add in new datatags for RNAi_series
    #     for rnai_n in RNAi_names:
    #         self.datatags.append(rnai_n)
    #
    #     self.outputs = []  # Storage array for all last timestep outputs
    #
    #     for ii, xxs in enumerate(xscales):  # Step through the x-scaling factors
    #
    #         if verbose is True:
    #             print('Run ', ii + 1, " of ", len(xscales))
    #
    #         model = PlanariaGRN1D(run_params, xscale=xxs)  # create a model using the specific parameters
    #
    #         # Run initialization of full model:
    #         model.initialize(knock_down=self.RNAi,
    #                          run_time=self.runtime,
    #                          run_time_step=self.runstep,
    #                          run_time_sample=self.runsample)
    #
    #         model.simulate(knock_down=self.RNAi,
    #                        run_time=self.runtime,
    #                        run_time_step=self.runstep,
    #                        run_time_sample=self.runsample)
    #
    #         data_dict = OrderedDict()
    #
    #         data_dict['bc_init'] = model.c_BC_time[-1]
    #         data_dict['wnt_init'] = model.c_WNT_time[-1]
    #         data_dict['notum_init'] = model.c_Notum_time[-1]
    #         data_dict['nrf_init'] = model.c_NRF_time[-1]
    #         data_dict['hh_init'] = model.c_HH_time[-1]
    #         data_dict['erk_init'] = model.c_ERK_time[-1]
    #         data_dict['apc_init'] = model.c_APC_time[-1]
    #         data_dict['camp_init'] = model.c_cAMP_time[-1]
    #
    #         data_dict['bc_sim'] = model.c_BC_sim_time[-1]
    #         data_dict['wnt_sim'] = model.c_WNT_sim_time[-1]
    #         data_dict['notum_sim'] = model.c_Notum_sim_time[-1]
    #         data_dict['nrf_sim'] = model.c_NRF_sim_time[-1]
    #         data_dict['hh_sim'] = model.c_HH_sim_time[-1]
    #         data_dict['erk_sim'] = model.c_ERK_sim_time[-1]
    #         data_dict['apc_sim'] = model.c_APC_sim_time[-1]
    #         data_dict['camp_sim'] = model.c_cAMP_sim_time[-1]
    #
    #         self.X = model.X
    #         self.Xscale = model.X / model.xmax
    #         self.seg_inds = model.seg_inds
    #
    #         for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):
    #
    #             if verbose is True:
    #                 print('Runing RNAi Sequence ', rnai_n)
    #
    #             model2 = PlanariaGRN(run_params, xscale=xxs)  # create a model using the specific parameters
    #
    #             # Run a new initialization of full model:
    #             model2.initialize(knock_down=self.RNAi,
    #                               run_time=self.runtime,
    #                               run_time_step=self.runstep,
    #                               run_time_sample=self.runsample)
    #
    #             if t_reinit > 0.0:
    #                 model2.reinitialize(knock_down=rnai_s,
    #                                     run_time=t_reinit,
    #                                     run_time_step=self.runstep,
    #                                     run_time_sample=self.runsample)
    #
    #             model2.simulate(knock_down=rnai_s,
    #                             run_time=self.runtime,
    #                             run_time_step=self.runstep,
    #                             run_time_sample=self.runsample)
    #
    #             # FIXME later we want this to be Markov Model output
    #             dic_lab_erk = rnai_n + '_outerk_sim'
    #             dic_lab_bc = rnai_n + '_outbc_sim'
    #             data_dict[dic_lab_erk] = model2.c_ERK_sim_time[-1] * 1
    #             data_dict[dic_lab_bc] = model2.c_BC_sim_time[-1] * 1
    #
    #         self.outputs.append(data_dict)
    #
    #         if plotwhilesolving:
    #
    #             for tag in self.datatags:
    #
    #                 if tag.endswith('sim'):
    #                     rtype = 'sim'
    #
    #                 else:
    #                     rtype = 'init'
    #
    #                 cj = data_dict[tag]
    #
    #                 impath = os.path.join(self.savepath, tag)
    #                 os.makedirs(impath, exist_ok=True)
    #
    #                 fnme = tag + '_Xscale' + str(xxs) + '.png'
    #
    #                 impath_c_init = os.path.join(impath, fnme)
    #
    #                 self.lineplot_1D(cj, impath_c_init, runtype=rtype)
    #
    #                 #                     plt.figure()
    #                 #                     ax = plt.subplot(111)
    #                 #                     plt.plot(self.X*1e3, cj, '-r', linewidth = 2.0)
    #                 #                     plt.xlabel('Distance [mm]')
    #                 #                     plt.ylabel('Concentration [mM/L]')
    #                 #                     plt.savefig(impath_c_init, format = 'png', dpi = 300)
    #                 #                     plt.close()

    def sim_rnai(self, run_params, RNAi_series, RNAi_names, verbose=True, run_time_init = 36000.0,
                 run_time_sim = 36000.0, run_time_step = 60, run_time_sample = 50, run_time_reinit = 12,
                 reset_clims = True, animate = False, plot = True, save_dir = 'SimRNAi_1'):


        # general saving directory for this procedure:
        self.savedir_sim_rnai = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_sim_rnai, exist_ok=True)


        # Create datatags for the harness to save data series to:
        self.datatags = []

        for tag in self.model.conc_tags:
            data_tagi = tag + '_init'
            data_tags = tag + '_sim'
            self.datatags.append(data_tagi)
            self.datatags.append(data_tags)

        # add in new datatags for RNAi_series
        for rnai_n in RNAi_names:

            for tag in self.model.conc_tags:
                data_tags = tag + rnai_n
                self.datatags.append(data_tags)

        self.data_dict = OrderedDict()

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

        # self.data_dict['base'] = copy.deepcopy(self.model.molecules_time)

        for tag in self.model.conc_tags:
            data_tagi = tag + '_init'
            data_tags = tag + '_sim'
            self.data_dict[data_tagi] = self.model.molecules_time[tag][-1]*1
            self.data_dict[data_tags] = self.model.molecules_sim_time[tag][-1]*1

        self.X = self.model.X
        self.Xscale = self.model.X / self.model.xmax
        self.seg_inds = self.model.seg_inds

        if plot:

            if verbose is True:
                print('Plotting base simulation...')

            fni = 'Triplot_init_'
            fns = 'Triplot_sim_'

            self.model.triplot(-1, plot_type='init', fname=fni, dirsave=self.savedir_sim_rnai,
                               cmaps=None, clims=None, autoscale=False)

            self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=self.savedir_sim_rnai,
                               cmaps=None, clims=None, autoscale=False)

        if animate:
            if verbose is True:
                print('Animating base simulation...')

            dni = 'TriplotAni_init'
            dns = 'TriplotAni_sim'

            savedir_anii = os.path.join(self.savedir_sim_rnai, dni)
            os.makedirs(savedir_anii, exist_ok=True)

            self.model.animate_triplot(ani_type='init', dirsave=savedir_anii,
                                       cmaps=None, clims=None, autoscale=False)

            savedir_anis = os.path.join(self.savedir_sim_rnai, dns)
            os.makedirs(savedir_anis, exist_ok=True)

            self.model.animate_triplot(ani_type='sim', dirsave=savedir_anis,
                                       cmaps=None, clims=None, autoscale=False)

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

            for tag in self.model.conc_tags:
                data_tags = tag + rnai_n
                self.data_dict[data_tags] = self.model.molecules_sim_time[tag][-1] * 1

            if plot:
                if verbose is True:
                    print('Plotting RNAi Sequence ', rnai_n, '...')

                fns = 'Triplot_'+ rnai_n + '_'

                self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=self.savedir_sim_rnai,
                                   cmaps=None, clims=None, autoscale=False)

            if animate:

                if verbose is True:
                    print('Animating RNAi Sequence ', rnai_n, '...')

                dns = 'TriplotAni_'+ rnai_n

                savedir_anis = os.path.join(self.savedir_sim_rnai, dns)

                os.makedirs(savedir_anis, exist_ok=True)

                self.model.animate_triplot(ani_type='sim', dirsave=savedir_anis,
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




