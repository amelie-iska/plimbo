#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
1D Simulator object for Planarian Interface for PLIMBO module.
'''

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from matplotlib import rcParams
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
from betse.util.type.mapping.mapcls import DynamicValue, DynamicValueDict
from betse.science.parameters import Parameters


class PlanariaGRN(object):
    """
    Object describing 1D version of core GRN model.
    A BETSE config file is used to define paths for saving image and data exports.

    """

    def __init__(self, config_filename, pdict, xscale=1.0, verbose = False):


        # BETSE parameters object:
        self.p = Parameters.make(config_filename)

        self.verbose = verbose

        self.RNAi_defaults = {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1,
                               'wnt': 1, 'hh': 1, 'camp': 1, 'dynein': 1}

        self.pdict = pdict

        self.x_scale = xscale

        self.make_mesh()

        self.prime_model()

        # Initialize the transport field and nerve density:
        self.load_transport_field()

        self.runtype = 'init'

        if self.verbose is True:
            print("-----------------------------")
            print("Successfully generated 1D model!")
            print("-----------------------------")


    def make_mesh(self):

        if self.verbose is True:
            print("Creating 1D mesh...")

        # X-axis
        self.xmin = 0.0
        self.xmax = 0.01 * self.x_scale
        self.xmid = (self.xmax - self.xmin) / 2
        self.dx = 1.0e-4 * self.x_scale
        self.nx = int((self.xmax - self.xmin) / self.dx)
        self.X = np.linspace(self.xmin, self.xmax, self.nx)

        # Cut points -- indicies of the x-axis to cut along:
        self.cut_points = np.array([0.002, 0.004, 0.006, 0.008]) * self.x_scale
        self.cut_inds, _ = self.find_nearest(self.X, self.cut_points)
        self.get_seg_inds()

        # build matrices
        self.build_matrices()


    def prime_model(self):


        if self.verbose is True:
            print("Initializing parameters and variables...")

        # tags for easy reference to concentrations of the model:
        self.conc_tags = ['β-Cat', 'Erk', 'Wnt', 'Hh', 'NRF', 'Notum', 'APC', 'cAMP']

        # Default RNAi keys:
        self.RNAi_defaults = {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1,
         'wnt': 1, 'hh': 1, 'camp': 1,'dynein': 1}

        self.init_plots()

        # Beta cat parameters
        self.r_bc = self.pdict['r_bc']
        self.d_bc = self.pdict['d_bc']
        self.d_bc_deg = self.pdict['d_bc_deg']
        self.K_bc_apc = self.pdict['K_bc_apc']
        self.n_bc_apc = self.pdict['n_bc_apc']
        self.K_bc_camp = self.pdict['K_bc_camp']
        self.n_bc_camp = self.pdict['n_bc_camp']

        self.c_BC = np.ones(self.nx)
        self.c_BC_time = []

        # ERK parameters
        self.r_erk = 5.0e-3
        self.d_erk = 5.0e-3
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.n_erk_bc = self.pdict['n_erk_bc']

        self.c_ERK = np.zeros(self.nx)
        self.c_ERK_time = []

        # APC parameters
        self.r_apc = 5.0e-3
        self.d_apc = 5.0e-3
        self.K_apc_wnt = self.pdict['K_apc_wnt']
        self.n_apc_wnt = self.pdict['n_apc_wnt']

        self.c_APC = np.zeros(self.nx)
        self.c_APC_time = []

        # Hedgehog parameters:
        self.r_hh = self.pdict['r_hh']
        self.d_hh = self.pdict['d_hh']
        self.D_hh = self.pdict['D_hh']
        #         self.u_hh = pdict['u_hh']

        self.c_HH = np.zeros(self.nx)
        self.c_HH_time = []

        # Wnt parameters
        self.r_wnt = self.pdict['r_wnt']
        self.d_wnt = self.pdict['d_wnt']
        self.d_wnt_deg = self.pdict['d_wnt_deg']
        self.D_wnt = self.pdict['D_wnt']
        self.K_wnt_notum = self.pdict['K_wnt_notum']
        self.n_wnt_notum = self.pdict['n_wnt_notum']
        self.K_wnt_hh = self.pdict['K_wnt_hh']
        self.n_wnt_hh = self.pdict['n_wnt_hh']
        self.K_wnt_camp = self.pdict['K_wnt_camp']
        self.n_wnt_camp = self.pdict['n_wnt_camp']

        self.c_WNT = np.zeros(self.nx)
        self.c_WNT_time = []

        # Notum regulating factor (NRF) parameters
        self.r_nrf = self.pdict['r_nrf']
        self.d_nrf = self.pdict['d_nrf']
        self.K_nrf_bc = self.pdict['K_nrf_bc']
        self.n_nrf_bc = self.pdict['n_nrf_bc']
        self.D_nrf = self.pdict['D_nrf']
        self.u_nrf = self.pdict['u_nrf']

        self.c_NRF = np.zeros(self.nx)
        self.c_NRF_time = []

        # Notum parameters
        self.D_notum = self.pdict['D_notum']
        self.r_notum = 5.0e-3
        self.d_notum = 5.0e-3
        self.K_notum_nrf = self.pdict['K_notum_nrf']
        self.n_notum_nrf = self.pdict['n_notum_nrf']

        self.c_Notum = np.zeros(self.nx)
        self.c_Notum_time = []

        # cAMP parameters
        self.r_camp = 5.0e-3
        self.d_camp = 5.0e-3

        self.c_cAMP = np.ones(self.nx) * 1.0

        self.Do = self.pdict['Do']

        # Short x-axis:
        self.Xs = np.dot(self.Mx, self.X)

    def load_transport_field(self):

        # Transport fields
        self.K_u = self.pdict['K_u']
        self.n_u = self.pdict['n_u']

        self.u = 1 / (1 + (self.Xs / (self.xmid / self.K_u)) ** self.n_u)

        self.NerveDensity = 1 / (1 + (self.X / (self.xmid / self.K_u)) ** self.n_u)

    def log_details(self):

        print("Spatial points: ", self.nx)
        print("Time points: ", self.nt)
        print("Sampled time points: ", len(self.tsample))

    def build_matrices(self):
        """
        Builds gradient and laplacian matrices for whole model
        """

        Gx = np.zeros((self.nx - 1, self.nx))  # gradient matrix
        Mx = np.zeros((self.nx - 1, self.nx))  # averaging matrix

        for i, pi in enumerate(self.X):

            if i != self.nx - 1:
                Gx[i, i + 1] = 1 / self.dx
                Gx[i, i] = -1 / self.dx

                Mx[i, i + 1] = 1 / 2
                Mx[i, i] = 1 / 2

        self.Lap = np.dot(-Gx.T, Gx)
        self.Lap_inv = np.linalg.pinv(self.Lap)
        self.Gx_inv = np.linalg.pinv(Gx)
        self.Gx = Gx * 1
        self.Mx = Mx * 1

        # build matrices for cut model:
        Gx_sim = Gx * 1
        Mx_sim = Mx * 1

        #         x_inds = np.linspace(0, model.nx -1, model.nx, dtype = np.int)

        for i, (a, b) in enumerate(self.seg_inds):

            if i < len(self.seg_inds) - 1:

                seg2 = self.seg_inds[i + 1]

                a2 = seg2[0]
                b2 = seg2[-1]

                Gx_sim[b - 1, a2] = 0.0
                Gx_sim[b - 1, b - 1] = 0.0

            elif a == 0:
                Gx_sim[b - 1, b] = 0.0
                Gx_sim[b - 1, b - 1] = 0.0

        self.Lap_sim = np.dot(-Gx_sim.T, Gx_sim)
        self.Lap_sim_inv = np.linalg.pinv(self.Lap_sim)
        self.Gx_sim_inv = np.linalg.pinv(Gx_sim)
        self.Gx_sim = Gx_sim * 1
        self.Mx_sim = Mx_sim * 1

    def get_gradient(self, conc, runtype):

        # Gradient and laplacian of concentration
        if runtype == 'sim':

            g_conc = np.dot(self.Gx_sim, conc)  # gradient of concentration
            m_conc = np.dot(self.Mx_sim, conc)  # mean of concentration

        else:
            g_conc = np.dot(self.Gx, conc)
            m_conc = np.dot(self.Mx, conc)  # mean of concentration

        return g_conc, m_conc

    def get_div(self, fx, runtype):

        if self.runtype == 'sim':

            div_fx = np.dot(-self.Gx_sim.T, fx)  # divergence of the flux

        else:
            div_fx = np.dot(-self.Gx.T, fx)

        return div_fx

    def find_nearest(self, aa, vals):
        """
        Returns nearest matches to vals in numpy array aa
        """
        inds = []
        for v in vals:
            idx = (np.abs(aa - v)).argmin()
            inds.append(idx)
        return inds, aa[inds]

    def get_seg_inds(self):

        seg_inds = []

        for i, j in enumerate(self.cut_inds):

            if i == 0:

                seg_inds.append([0, j])

            elif i == len(self.cut_inds) - 1:

                seg_inds.append([self.cut_inds[i - 1], self.cut_inds[i]])

                seg_inds.append([self.cut_inds[i], self.nx])

            else:

                seg_inds.append([self.cut_inds[i - 1], self.cut_inds[i]])

        self.seg_inds = np.asarray(seg_inds)

    def segmentation(self, aa):
        """
        Segmentation of a model array property, aa.

        """
        segments = []
        for a, b in self.seg_inds:
            segments.append(aa[a:b])

        return np.asarray(segments)

    def update_bc(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in beta-cat levels in space and time.
        """

        # Growth and decay
        iAPC = (self.c_APC / self.K_bc_apc) ** self.n_bc_apc
        term_apc = iAPC / (1 + iAPC)

        icAMP = (self.c_cAMP / self.K_bc_camp) ** self.n_bc_camp
        term_camp = 1 / (1 + icAMP)

        # gradient and midpoint mean concentration:
        g_bc, m_bc = self.get_gradient(self.c_BC, self.runtype)

        # Motor transport term:
        #         conv_term = m_bc*self.u*self.u_bc*kinesin

        #         flux = -g_bc*self.D_bc + conv_term
        flux = -g_bc * self.Do

        # divergence of the flux
        div_flux = self.get_div(flux, self.runtype)

        # change of bc:
        del_bc = (-div_flux + rnai * self.r_bc * self.NerveDensity -
                  self.d_bc * self.c_BC - self.d_bc_deg * term_apc * self.c_BC * term_camp)

        return del_bc  # change in bc

    def update_nrf(self, dynein=1.0):
        """
        Method describing change in NRF levels in space and time.
        """

        # Growth and decay interactions:
        iBC = (self.c_BC / self.K_nrf_bc) ** self.n_nrf_bc

        term_bc = iBC / (1 + iBC)

        # Gradient and laplacian of concentration
        g_nrf, m_nrf = self.get_gradient(self.c_NRF, self.runtype)

        # Motor transport term:
        conv_term = m_nrf * self.u * self.u_nrf * dynein

        flux = -g_nrf * self.D_nrf + conv_term

        div_flux = self.get_div(flux, self.runtype)

        # divergence of flux, growth and decay, breakdown in chemical tagging reaction:
        del_nrf = (-div_flux + self.r_nrf * term_bc - self.d_nrf * self.c_NRF)

        return del_nrf  # change in NRF

    def update_notum(self, rnai=1.0):
        """
        Method describing change in Notum levels in space and time.
        """

        iNRF = (self.c_NRF / self.K_notum_nrf) ** self.n_notum_nrf

        term_nrf = iNRF / (1 + iNRF)

        # Gradient and laplacian of concentration
        g_not, m_not = self.get_gradient(self.c_Notum, self.runtype)

        flux = -g_not * self.D_notum

        div_flux = self.get_div(flux, self.runtype)

        del_notum = -div_flux + rnai * self.r_notum * term_nrf - self.d_notum * self.c_Notum

        return del_notum

    def update_wnt(self, rnai=1.0):
        """
        Method describing change in Wnt1 and Wnt 11 levels in space and time.
        """

        # Growth and decay
        iNotum = (self.c_Notum / self.K_wnt_notum) ** self.n_wnt_notum
        iHH = (self.c_HH / self.K_wnt_hh) ** self.n_wnt_hh
        icAMP = (self.c_cAMP / self.K_wnt_camp) ** self.n_wnt_camp

        term_hh = iHH / (1 + iHH)
        term_notum = iNotum / (1 + iNotum)
        term_camp = icAMP / (1 + icAMP)

        # Gradient and mean of concentration
        g_wnt, m_wnt = self.get_gradient(self.c_WNT, self.runtype)

        # Motor transport term:
        flux = -self.D_wnt * g_wnt

        # divergence
        div_flux = self.get_div(flux, self.runtype)

        del_wnt = (-div_flux + rnai * self.r_wnt * term_hh * term_camp * self.NerveDensity -
                   self.d_wnt * self.c_WNT - self.d_wnt_deg * term_notum * self.c_WNT)

        return del_wnt  # change in Wnt

    def update_hh(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in Hedgehog levels in space and time.
        """

        # Gradient and mean of concentration
        g_hh, m_hh = self.get_gradient(self.c_HH, self.runtype)

        #         Motor transport term:
        #         conv_term = m_hh*self.u*self.u_hh*kinesin

        #         flux = -g_hh*self.D_hh + conv_term
        flux = -g_hh * self.D_hh

        #         divergence
        div_flux = self.get_div(flux, self.runtype)

        # final change in hh
        del_hh = (-div_flux + rnai * self.r_hh * self.NerveDensity - self.d_hh * self.c_HH)

        return del_hh  # change in Hedgehog

    def update_erk(self, rnai=1.0):
        """
        Method describing change in ERK levels in space and time.
        """

        iBC = (self.c_BC / self.K_erk_bc) ** self.n_erk_bc

        term_bc = 1 / (1 + iBC)

        # Gradient and mean of concentration
        g_erk, m_erk = self.get_gradient(self.c_ERK, self.runtype)

        # Motor transport term:
        flux = -self.Do * g_erk

        # divergence
        div_flux = self.get_div(flux, self.runtype)

        del_erk = -div_flux + rnai * self.r_erk * term_bc - self.d_erk * self.c_ERK

        return del_erk

    def update_apc(self, rnai=1.0):
        """
        Method describing change in APC levels in space and time.
        """

        iWNT = (self.c_WNT / self.K_apc_wnt) ** self.n_apc_wnt

        term_wnt = 1 / (1 + iWNT)

        # Gradient and mean of concentration
        g_apc, m_apc = self.get_gradient(self.c_APC, self.runtype)

        # Motor transport term:
        flux = -self.Do * g_apc

        # divergence
        div_flux = self.get_div(flux, self.runtype)

        del_apc = -div_flux + rnai * self.r_apc * term_wnt - self.d_apc * self.c_APC

        return del_apc

    def update_camp(self, rnai=1.0):
        """
        Method describing change in cAMP levels in space and time.
        """

        del_cAMP = rnai * self.r_camp - self.d_camp * self.c_cAMP

        return del_cAMP

    def clear_cache_init(self):

        molecules = OrderedDict({})

        self.c_BC_time = []
        self.c_ERK_time = []
        self.c_WNT_time = []
        self.c_HH_time = []
        self.c_NRF_time = []
        self.c_Notum_time = []
        self.c_APC_time = []
        self.c_cAMP_time = []

        self.delta_ERK_time = []

        molecules['β-Cat'] = DynamicValue(
            lambda: self.c_BC_time, lambda value: self.__setattr__('c_BC_time', value))

        molecules['Erk'] = DynamicValue(
            lambda: self.c_ERK_time, lambda value: self.__setattr__('c_ERK_time', value))

        molecules['Wnt'] = DynamicValue(
            lambda: self.c_WNT_time, lambda value: self.__setattr__('c_WNT_time', value))

        molecules['Hh'] = DynamicValue(
            lambda: self.c_HH_time, lambda value: self.__setattr__('c_HH_time', value))

        molecules['NRF'] = DynamicValue(
            lambda: self.c_NRF_time, lambda value: self.__setattr__('c_NRF_time', value))

        molecules['Notum'] = DynamicValue(
            lambda: self.c_Notum_time, lambda value: self.__setattr__('c_Notum_time', value))

        molecules['APC'] = DynamicValue(
            lambda: self.c_APC_time, lambda value: self.__setattr__('c_APC_time', value))

        molecules['cAMP'] = DynamicValue(
            lambda: self.c_cAMP_time, lambda value: self.__setattr__('c_cAMP_time', value))

        self.molecules_time = DynamicValueDict(molecules)

    def clear_cache_reinit(self):

        molecules = OrderedDict({})

        self.c_BC_time2 = []
        self.c_ERK_time2 = []
        self.c_WNT_time2 = []
        self.c_HH_time2 = []
        self.c_NRF_time2 = []
        self.c_Notum_time2 = []
        self.c_APC_time2 = []
        self.c_cAMP_time2 = []

        self.delta_ERK_time2 = []

        molecules['β-Cat'] = DynamicValue(
            lambda: self.c_BC_time2, lambda value: self.__setattr__('c_BC_time2', value))

        molecules['Erk'] = DynamicValue(
            lambda: self.c_ERK_time2, lambda value: self.__setattr__('c_ERK_time2', value))

        molecules['Wnt'] = DynamicValue(
            lambda: self.c_WNT_time2, lambda value: self.__setattr__('c_WNT_time2', value))

        molecules['Hh'] = DynamicValue(
            lambda: self.c_HH_time2, lambda value: self.__setattr__('c_HH_time2', value))

        molecules['NRF'] = DynamicValue(
            lambda: self.c_NRF_time2, lambda value: self.__setattr__('c_NRF_time2', value))

        molecules['Notum'] = DynamicValue(
            lambda: self.c_Notum_time2, lambda value: self.__setattr__('c_Notum_time2', value))

        molecules['APC'] = DynamicValue(
            lambda: self.c_APC_time2, lambda value: self.__setattr__('c_APC_time2', value))

        molecules['cAMP'] = DynamicValue(
            lambda: self.c_cAMP_time2, lambda value: self.__setattr__('c_cAMP_time2', value))

        self.molecules_time2 = DynamicValueDict(molecules)

    def clear_cache_sim(self):

        molecules = OrderedDict({})

        self.c_BC_sim_time = []
        self.c_ERK_sim_time = []
        self.c_WNT_sim_time = []
        self.c_HH_sim_time = []
        self.c_NRF_sim_time = []
        self.c_Notum_sim_time = []
        self.c_APC_sim_time = []
        self.c_cAMP_sim_time = []

        self.delta_ERK_sim_time = []

        molecules['β-Cat'] = DynamicValue(
            lambda: self.c_BC_sim_time, lambda value: self.__setattr__('c_BC_sim_time', value))

        molecules['Erk'] = DynamicValue(
            lambda: self.c_ERK_sim_time, lambda value: self.__setattr__('c_ERK_sim_time', value))

        molecules['Wnt'] = DynamicValue(
            lambda: self.c_WNT_sim_time, lambda value: self.__setattr__('c_WNT_sim_time', value))

        molecules['Hh'] = DynamicValue(
            lambda: self.c_HH_sim_time, lambda value: self.__setattr__('c_HH_sim_time', value))

        molecules['NRF'] = DynamicValue(
            lambda: self.c_NRF_sim_time, lambda value: self.__setattr__('c_NRF_sim_time', value))

        molecules['Notum'] = DynamicValue(
            lambda: self.c_Notum_sim_time, lambda value: self.__setattr__('c_Notum_sim_time', value))

        molecules['APC'] = DynamicValue(
            lambda: self.c_APC_sim_time, lambda value: self.__setattr__('c_APC_sim_time', value))

        molecules['cAMP'] = DynamicValue(
            lambda: self.c_cAMP_sim_time, lambda value: self.__setattr__('c_cAMP_sim_time', value))


        self.molecules_sim_time = DynamicValueDict(molecules)

    def run_loop(self,
                 knockdown=None):

        for tt in self.time:

            delta_bc = self.update_bc(rnai=knockdown['bc']) * self.dt  # time update beta-catenin
            delta_wnt = self.update_wnt(rnai=knockdown['wnt']) * self.dt  # time update wnt
            delta_hh = self.update_hh(rnai=knockdown['hh']) * self.dt  # time update hh
            delta_nrf = self.update_nrf(dynein=knockdown['dynein']) * self.dt  # update NRF
            delta_notum = self.update_notum(rnai=knockdown['notum']) * self.dt  # time update Notum
            delta_erk = self.update_erk(rnai=knockdown['erk']) * self.dt  # time update ERK
            delta_apc = self.update_apc(rnai=knockdown['apc']) * self.dt  # time update APC
            delta_camp = self.update_camp(rnai=knockdown['camp']) * self.dt  # time update cAMP

            self.c_BC += delta_bc  # time update beta-catenin
            self.c_WNT += delta_wnt  # time update Wnt
            self.c_HH += delta_hh  # time update Hh
            self.c_NRF += delta_nrf  # time update NRF
            self.c_Notum += delta_notum  # time update Notum
            self.c_ERK += delta_erk  # time update ERK
            self.c_APC += delta_apc  # time update APC
            self.c_cAMP += delta_camp  # time update cAMP

            if tt in self.tsample:

                if self.runtype == 'init':

                    self.c_BC_time.append(self.c_BC * 1)
                    self.c_WNT_time.append(self.c_WNT * 1)
                    self.c_HH_time.append(self.c_HH * 1)
                    self.c_Notum_time.append(self.c_Notum * 1)
                    self.c_NRF_time.append(self.c_NRF * 1)
                    self.c_ERK_time.append(self.c_ERK * 1)
                    self.c_APC_time.append(self.c_APC * 1)
                    self.c_cAMP_time.append(self.c_cAMP * 1)

                    self.delta_ERK_time.append(delta_erk.mean() * 1)

                elif self.runtype == 'reinit':

                    self.c_BC_time2.append(self.c_BC * 1)
                    self.c_WNT_time2.append(self.c_WNT * 1)
                    self.c_HH_time2.append(self.c_HH * 1)
                    self.c_Notum_time2.append(self.c_Notum * 1)
                    self.c_NRF_time2.append(self.c_NRF * 1)
                    self.c_ERK_time2.append(self.c_ERK * 1)
                    self.c_APC_time2.append(self.c_APC * 1)
                    self.c_cAMP_time2.append(self.c_cAMP * 1)

                    self.delta_ERK_time2.append(delta_erk.mean() * 1)


                elif self.runtype == 'sim':

                    self.c_BC_sim_time.append(self.c_BC * 1)
                    self.c_WNT_sim_time.append(self.c_WNT * 1)
                    self.c_HH_sim_time.append(self.c_HH * 1)
                    self.c_Notum_sim_time.append(self.c_Notum * 1)
                    self.c_NRF_sim_time.append(self.c_NRF * 1)
                    self.c_ERK_sim_time.append(self.c_ERK * 1)
                    self.c_APC_sim_time.append(self.c_APC * 1)
                    self.c_cAMP_sim_time.append(self.c_cAMP * 1)

                    self.delta_ERK_sim_time.append(delta_erk.mean() * 1)

    def initialize(self,
                   knockdown= None,
                   run_time=48.0 * 3600,
                   run_time_step=10,
                   run_time_sample=100):

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = run_time_sample
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_init()
        self.runtype = 'init'
        self.run_loop(knockdown=knockdown)
        self.tsample_init = self.tsample

        if self.verbose:
            print("-----------------------------")
            print("Successfully completed init of 1D model!")
            print("-----------------------------")

    def reinitialize(self,
                     knockdown=None,
                     run_time=48.0 * 3600,
                     run_time_step=10,
                     run_time_sample=100):

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = run_time_sample
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_reinit()
        self.runtype = 'reinit'
        self.run_loop(knockdown=knockdown)
        self.tsample_reinit = self.tsample

        if self.verbose:
            print("-----------------------------")
            print("Successfully completed reinit of 1D model!")
            print("-----------------------------")

    def simulate(self,
                 knockdown=None,
                 run_time=48.0 * 3600,
                 run_time_step=10,
                 run_time_sample=100):

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = run_time_sample
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_sim()
        self.runtype = 'sim'
        self.run_loop(knockdown=knockdown)

        self.tsample_sim = self.tsample

        if self.verbose:
            print("-----------------------------")
            print("Successfully completed sim of 1D model!")
            print("-----------------------------")

    def init_plots(self):

        # default plot legend scaling (can be modified)
        mol_clims = OrderedDict()

        mol_clims['β-Cat'] = (0, 100.0)
        mol_clims['Erk'] = (0, 1.0)
        mol_clims['Wnt'] = (0, 200.0)
        mol_clims['Hh'] = (0, 650.0)
        mol_clims['NRF'] = (0, 3000.0)
        mol_clims['Notum'] = (0, 1.0)
        mol_clims['APC'] = (0, 1.0)
        mol_clims['cAMP'] = (0, 1.0)

        self.default_clims = mol_clims

        mol_cmaps = OrderedDict()

        mol_cmaps['β-Cat'] = 'Blue'
        mol_cmaps['Erk'] = 'Red'
        mol_cmaps['Wnt'] = 'DodgerBlue'
        mol_cmaps['Hh'] = 'DarkCyan'
        mol_cmaps['NRF'] = 'Blue'
        mol_cmaps['Notum'] = 'Green'
        mol_cmaps['APC'] = 'OrangeRed'
        mol_cmaps['cAMP'] = 'DeepSkyBlue'

        self.default_cmaps = mol_cmaps

    def triplot(self, ti, plot_type = 'init', dirsave = 'Triplot', reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12)):


        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]

            fstr = 'Triplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]

            fstr = 'Triplot2_'+ str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]
            carray3 = self.molecules_sim_time['Notum'][ti]

            fstr = 'Triplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.sim_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, sharex=True, figsize=fsize)

        axarr[0].plot(self.X/self.xmax, carray1, color=cmaps['Erk'], linewidth=linew)
        axarr[0].set_title("ERK")
        axarr[0].set_ylabel('Concentration [nM]')

        axarr[1].plot(self.X /self.xmax, carray2, color=cmaps['β-Cat'], linewidth=linew)
        axarr[1].set_title("beta-cat")
        axarr[1].set_ylabel('Concentration [nM]')

        axarr[2].plot(self.X /self.xmax, carray3, color=cmaps['Notum'], linewidth=linew)
        axarr[2].set_title("Notum")
        axarr[2].set_ylabel('Concentration [nM]')

        axarr[2].set_xlabel('Normalized Axis Distance')

        fig.subplots_adjust(hspace=0.15)
        fig.suptitle('Initialization', x=0.1, y=0.94)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string)

        plt.savefig(fname, format='png', dpi=reso)
        plt.close()

    def biplot(self, ti, plot_type = 'init', dirsave = 'Biplot', reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12)):


        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]

            fstr = 'Biplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]

            fstr = 'Biplot2_'+ str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim

            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]

            fstr = 'Biplot_sim_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.sim_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(2, sharex=True, figsize=fsize)

        axarr[0].plot(self.X/self.xmax, carray1, color=cmaps['Erk'], linewidth=linew)
        axarr[0].set_title("ERK")
        axarr[0].set_ylabel('Concentration [nM]')

        axarr[1].plot(self.X /self.xmax, carray2, color=cmaps['β-Cat'], linewidth=linew)
        axarr[1].set_title("beta-cat")
        axarr[1].set_ylabel('Concentration [nM]')

        axarr[1].set_xlabel('Normalized Axis Distance')

        fig.subplots_adjust(hspace=0.15)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.1, y=0.94)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def plot(self, ti, ctag, plot_type='init', auto_clim=True, fsave = 'Plot', reso = 150, linew = 3.0,
                cmaps=None, fontsize=16.0, fsize=(4, 10)):


        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':
            tsample = self.tsample_init
            carray = self.molecules_time[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, fsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':
            tsample = self.tsample_reinit
            carray = self.molecules_time2[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, fsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray = self.molecules_sim_time[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.sim_export_dirname, fsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})
        plt.figure(figsize=fsize)

        plt.plot(self.X / self.xmax, carray, color=cmaps[ctag], linewidth=linew)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = 'Time: ' + str(tdays) + ' Hours'
        plt.title(tit_string)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()




