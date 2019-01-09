#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Top-level Planarian Interface for Modelling Body Organization (PLIMBO) module.
'''

import os
import os.path

from plimbo.harness import ModelHarness


class PlimboRunner(object):
    """
    Top level interface for running all functionality of the PLIMBO simulator.

    """

    def __init__(self, fn_config, verbose = True, head_frags = None, tail_frags = None):
        """

        :param fn_config: BETSE config file for model creation and file saving information
        """

        self.fn_config = fn_config # assign path to BETSE config file
        self.verbose = verbose

        # specify fragments that are heads or tails for the Markov simulation:
        if head_frags is None:
            self.head_frags = [0]
        else:
            self.head_frags = head_frags
        if tail_frags is None:
            self.tail_frags = [4]
        else:
            self.tail_frags = tail_frags


    def simRNAi(self, RNAi_vect = None, RNAi_tags = None, params = None, run_time_init = 36000.0,
                run_time_sim = 36000.0, run_time_step = 60.0, run_time_sample = 50.0, plot_frags = True,
                run_time_reinit = 12.0,  xscale = 1.0, verbose = True, new_mesh = False,
                save_dir = 'SimRNAi_1', reset_clims = True, animate = False, plot = True, axisoff = False,
                plot_type = 'Triplot', ani_type = 'Triplot', save_harness = False, harness_type='1D',
                fsize = (6,8), clims = None):


        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo = params, xscale = xscale, harness_type = '1D',
                                  verbose = verbose, new_mesh=new_mesh, savedir = 'Harness1D',
                                  plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo = params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D',
                                  plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        else:
            print("Harness type can only be '1D' or '2D'.") # FIXME raise proper exception


        # run a single set of parameters with RNAi testing sequence:
        master.run_simRNAi(RNAi_series = RNAi_vect,
                           RNAi_names = RNAi_tags, verbose=verbose,
                           run_time_init = run_time_init,
                           run_time_sim = run_time_sim,
                           run_time_step = run_time_step,
                           run_time_sample = run_time_sample,
                           run_time_reinit = run_time_reinit,
                           reset_clims = reset_clims,
                           animate = animate,
                           plot = plot,
                           save_all=save_harness,
                           plot_type = plot_type,
                           save_dir = save_dir,
                           ani_type=ani_type,
                           axisoff=axisoff,
                           fsize = fsize,
                           clims=clims)

        self.master = master


    def sensitivity(self, params=None, run_time_init=36000.0, factor = 0.25, plot_frags = True,
                run_time_sim=36000.0, run_time_step=60.0, run_time_sample=50.0, run_type = 'init',
                 xscale=1.0, verbose=True, new_mesh=False, ani_type = 'Triplot',
                save_dir='Sensitivity_1', reset_clims=True, animate=False, plot=True, axisoff = False,
                plot_type='Triplot', save_harness=False, harness_type='1D', reference = ['Head','Tail'], fsize = (6,8),
                    clims = None, paramo_units = None):

        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception


        # run a sensitivity analysis on the model:
        master.run_sensitivity(factor = factor, run_type = run_type, verbose=verbose, run_time_init = run_time_init,
                        run_time_sim = run_time_sim, run_time_step = run_time_step, save_all = save_harness,
                        run_time_sample = run_time_sample, reset_clims = reset_clims, ani_type = ani_type,
                        animate = animate, plot = plot, save_dir = save_dir, plot_type = plot_type,
                        reference = reference, axisoff=axisoff, fsize=fsize, clims =clims, paramo_units=paramo_units)

        self.master = master



    def scaleRNAi(self, params=None, xscales = None,  RNAi_vect = None, RNAi_tags = None, plot_frags = True,
                  run_time_init=36000.0, run_time_sim=36000.0, run_time_step=60.0, run_time_sample=50.0,
               run_time_reinit=12.0, xscale=1.0, verbose=True, new_mesh=False, ani_type = 'Triplot',
               save_dir='ScaleRNAi_1', reset_clims=True, animate=False, plot=True, axisoff = False,
               plot_type='Triplot', save_harness=False, harness_type='1D', fsize = (6,8), clims = None):

        # Create an instance of the model harness:

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception


        # run a single set of parameters, scaling the model to various factors, with RNAi testing:
        master.run_scaleRNAi(xscales = xscales, RNAi_series = RNAi_vect, RNAi_names = RNAi_tags,
                               verbose=verbose, run_time_reinit=run_time_reinit, run_time_init=run_time_init,
                               run_time_sim=run_time_sim, run_time_step=run_time_step, save_all = save_harness,
                             run_time_sample=run_time_sample, reset_clims=reset_clims, ani_type = ani_type,
                               plot=plot, animate=animate, save_dir=save_dir, plot_type = plot_type, axisoff=axisoff,
                                fsize = fsize, clims = clims)

        self.master = master


    def searchRNAi(self, params=None, free_params = None, RNAi_vect = None, RNAi_tags = None,
                   run_time_init = 36000.0, axisoff = False,
                run_time_sim = 36000.0, run_time_step = 60.0, run_time_sample = 50.0, search_style = 'log',
                   factor = 0.8, levels = 1, run_time_reinit = 12.0, xscale = 1.0, verbose = True, new_mesh = False,
                save_dir = 'SearchRNAi_1', reset_clims = True, animate = False, plot = True, ani_type = 'Triplot',
                plot_type = 'Triplot', save_harness = True, harness_type='1D', plot_frags = True, fsize = (6,8),
                   clims = None, up_only = False):

        # Create an instance of the model harness:
        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='1D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness1D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo=params, xscale=xscale, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D', plot_frags = plot_frags,
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        else:
            print("Harness type can only be '1D' or '2D'.")  # FIXME raise proper exception

        # Run a local search on the model, changing each parameter progressively and running RNAi
        # set on the itteration:
        master.run_searchRNAi(RNAi_series = RNAi_vect, RNAi_names = RNAi_tags, factor = factor, levels = levels,
                                 search_style = search_style, verbose=verbose, run_time_reinit=run_time_reinit,
                                  run_time_init=run_time_init, run_time_sim=run_time_sim, save_all = save_harness,
                                 run_time_step=run_time_step, run_time_sample=run_time_sample, axisoff=axisoff,
                                  reset_clims=reset_clims, plot=plot, plot_type = plot_type, ani_type = ani_type,
                               animate=animate, save_dir=save_dir, free_params=free_params, fsize=fsize,
                              clims = clims, up_only=up_only)

        self.master = master


    def after_plot(self, loadpath, save_dir = 'Plots', plot_type = 'Triplot', output_type = 'sim', axisoff = False,
                   autoscale = False, clims=None, cmaps=None, verbose = True, harness_type = '1D', fsize = (6,8)):

        if verbose:
            print("Plotting simulation...")

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, harness_type='1D', plot_frags = True,
                                  verbose=verbose,  savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, harness_type='2D', plot_frags = True,
                                  verbose=verbose, new_mesh=False, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")

        master.plot_all_output(loadpath, save_dir = save_dir, plot_type=plot_type, output_type=output_type,
                               autoscale = autoscale, clims = clims, cmaps = cmaps, axisoff=axisoff, fsize=fsize)

        if verbose:
            print("Plotting completed.")

    def after_ani(self, loadpath, save_dir = 'Animations', ani_type = 'Triplot', output_type = 'sim', axisoff = False,
                  autoscale=False, clims=None, cmaps=None, verbose = True, harness_type = '1D', fsize = (6,8)):

        if verbose:
            print("Animating simulation...")

        if harness_type == '1D':

            master = ModelHarness(self.fn_config,  harness_type='1D', plot_frags = False,
                                  verbose=verbose, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, harness_type='2D', plot_frags = False,
                                  verbose=verbose, new_mesh=False, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")

        master.ani_all_output(loadpath, save_dir = save_dir, ani_type=ani_type, output_type=output_type,
                              autoscale=autoscale, clims=clims, cmaps=cmaps, axisoff = axisoff, fsize=fsize)

        if verbose:
            print("Animations completed.")

    def after_data(self, loadpath, substance ='Erk', save_dir = 'OutputData',
                   output_type = 'init', verbose = True, harness_type = '1D'):

        if verbose:
            print("Creating summary table exports for simulations...")

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, harness_type='1D', plot_frags = False,
                                  verbose=verbose, savedir='Harness1D')

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, harness_type='2D', plot_frags = False,
                                  verbose=verbose, new_mesh=False, savedir='Harness2D')

        else:
            print("Harness type can only be '1D' or '2D'.")

        load_fname = os.path.join(loadpath, "Master.gz")
        master = master.load(load_fname)

        savedir = os.path.join(master.savepath, save_dir)
        os.makedirs(savedir, exist_ok=True)

        master.output_delta_table(substance=substance,
                                  run_type=output_type, save_dir=savedir)

        master.output_summary_table(substance=substance,
                                  run_type=output_type, save_dir=savedir)

        if verbose:
            print("Exported summary data tables.")

    def frag_plot(self, loadpath, params = None, new_mesh = False, fsize = (6,8),
                  save_dir = 'Fragments', verbose = True, harness_type = '2D'):

        if harness_type == '1D':

            master = ModelHarness(self.fn_config, paramo = params, harness_type = '1D',
                                  verbose = verbose, new_mesh=new_mesh, savedir = 'Harness1D',
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        elif harness_type == '2D':

            master = ModelHarness(self.fn_config, paramo = params, harness_type='2D',
                                  verbose=verbose, new_mesh=new_mesh, savedir='Harness2D',
                                  head_frags=self.head_frags, tail_frags=self.tail_frags)

        else:
            print("Harness type can only be '1D' or '2D'.") # FIXME raise proper exception

        master.view_fragments(fsize=fsize)



