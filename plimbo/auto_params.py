#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Auto-parameter generator for sensitivity analysis and automatic model searching in PLIMBO module.
Works for both 1D and 2D simulation types.
'''

import numpy as np


class ParamsManager(object):
    """
    Object creating sets of procedurally-generated model parameters.

    """

    def __init__(self, param_o, levels=1, free_params=None,
                 N_runs_max=10.0):

        self.paramo_dict = param_o  # base parameters as a dictionary

        self.paramo_array = np.array(list(param_o.values()))  # base parameters as an array
        self.param_labels = np.array(list(param_o.keys()))  # Names of parameters as an array

    def create_sensitivity_matrix(self, factor = 0.1):

        self.factor = factor

        # array of percent changes to each parameter
        self.delta_vect = self.factor * self.paramo_array

        # diagonal array of delta factors for each parameter
        self.delta_M = np.diag(self.delta_vect)

        params_M = [self.paramo_array]

        for i in range(len(self.paramo_array)):
            params_M.append(self.paramo_array + self.delta_M[i, :])

        self.params_M = np.asarray(params_M)

        self.N_runs = self.params_M.shape[0]
        self.N_params = self.params_M.shape[1]

    def create_search_matrix(self, factor =0.8, levels = 1, style='log',
                             free_params = None, up_only = False):

        self.factor = factor
        self.levels = levels

        if style == 'log':

            self.delta_vect = (self.factor - 1) * np.ones(len(self.paramo_array))
            self.delta_M = np.diag(self.delta_vect) + 1

            params_M = [self.paramo_array]

            if free_params is None: # if all parameters are varied, add each perturbation to the final params matrix:

                for i in range(len(self.paramo_array)):

                    if up_only is False:

                        for j in range(self.levels):
                            params_M.append(self.paramo_array * (self.delta_M[i, :] ** (self.levels - j)))

                    for j in range(self.levels):
                        params_M.append(self.paramo_array * (1 / self.delta_M[i, :]) ** (j + 1))

            else:  # if some parameters are fixed, only add free parameters cases to the final params matrix:

                free_params_list = np.array(list(free_params.values()))

                for i in range(len(self.paramo_array)):

                    param_flag = free_params_list[i] # get the flag for the parameter

                    if param_flag == 1: # if the flag is True/1 then add this case to the params Matrix

                        if up_only is False:

                            for j in range(self.levels):
                                params_M.append(self.paramo_array * (self.delta_M[i, :] ** (self.levels - j)))

                        for j in range(self.levels):
                            params_M.append(self.paramo_array * (1 / self.delta_M[i, :]) ** (j + 1))



        elif style == 'lin':

            # array of percent changes to each parameter
            self.delta_vect = self.factor * self.paramo_array

            # diagonal array of delta factors for each parameter
            self.delta_M = np.diag(self.delta_vect)

            params_M = [self.paramo_array]

            if free_params is None: # if all parameters are varied, add each perturbation to the final params matrix:

                for i in range(len(self.paramo_array)):

                    if up_only is False:

                        for j in range(self.levels):
                            params_M.append(self.paramo_array - (self.levels - j) * self.delta_M[i, :])

                    for j in range(self.levels):
                        params_M.append(self.paramo_array + (j + 1) * self.delta_M[i, :])

            else: # if some parameters are fixed, only add free parameters cases to the final params matrix:

                free_params_list = np.array(list(free_params.values()))

                for i in range(len(self.paramo_array)):

                    param_flag = free_params_list[i] # get the flag for the parameter

                    if param_flag == 1: # if the flag is True/1 then add this case to the params Matrix

                        if up_only is False:

                            for j in range(self.levels):
                                params_M.append(self.paramo_array - (self.levels - j) * self.delta_M[i, :])

                        for j in range(self.levels):
                            params_M.append(self.paramo_array + (j + 1) * self.delta_M[i, :])

        else:
            print("Error! The choices for search matrix style are 'log' and 'lin'")

        self.params_M = np.asarray(params_M)
        self.N_runs = self.params_M.shape[0]
        self.N_params = self.params_M.shape[1]

    def create_random_matrix(self, factor = 0.8, levels = 1, free_params = None, style='log'):

        self.factor = factor
        self.levels = levels
        self.free_params = free_params

        params_at_levels = []

        if self.free_params is not None:

            for para, fi in zip(self.paramo_dict.values(), self.free_params.values()):

                if fi:

                    if style == 'lin':

                        pv = np.hstack((np.linspace(para * (1 - self.levels * self.factor),
                                                    para * (1 + self.levels * self.factor), 2 * self.levels), para))

                    elif style == 'log':

                        lowv = np.log(para) + self.levels * np.log(self.factor)
                        highv = np.log(para) - self.levels * np.log(self.factor)

                        log_pv = np.linspace(lowv, highv, 2 * self.levels)

                        pv = np.hstack((np.exp(log_pv), para))

                    else:
                        print("Error! Valid style choices are 'lin' and 'log'.")

                else:

                    pv = np.ones(2 * self.levels + 1) * para

                params_at_levels.append(pv * 1)

        else:  # else if free parameters aren't mentioned, alter all of them:

            for para in self.paramo_dict.values():

                if style == 'lin':

                    pv = np.hstack((np.linspace(para * (1 - self.levels * self.factor),
                                                para * (1 + self.levels * self.factor), 2 * self.levels), para))

                elif style == 'log':

                    lowv = np.log(para) + self.levels * np.log(self.factor)
                    highv = np.log(para) - self.levels * np.log(self.factor)

                    log_pv = np.linspace(lowv, highv, 2 * self.levels)

                    pv = np.hstack((np.exp(log_pv), para))

                else:
                    print("Error! Valid style choices are 'lin' and 'log'.")

                params_at_levels.append(pv * 1)

        unique_combos = set()  # define a set that will hold unique combinations
        for i in range(self.N_runs_max):
            para_list = []

            for pi_choices in params_at_levels:  # for each parameters vector

                pi_shuffled = np.random.permutation(pi_choices)  # shuffle it
                para_list.append(pi_shuffled[0])  # append the first value to the list

            unique_combos.add(tuple(para_list))  # add the set of parameters to the combos set

        params_M = [self.paramo_array]

        for paraset in unique_combos:
            params_M.append(list(paraset))

        self.params_at_levels = np.asarray(params_at_levels)

        self.params_M = np.asarray(params_M)
        self.N_runs = self.params_M.shape[0]
        self.N_params = self.params_M.shape[1]
