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
# from betse.util.type.mapping.mapcls import DynamicValue, DynamicValueDict
from betse.science.parameters import Parameters


class PlanariaGRN1D(object):
    """
    Object describing 1D version of core GRN model.
    A BETSE config file is used to define paths for saving image and data exports.

    """

    def __init__(self, *args, **kwargs):

        self.model_init(*args, **kwargs)

    def model_init(self, config_filename, pdict = None, xscale=1.0, new_mesh=False,
                 verbose=False):

        if pdict is None: # default parameters
            self.pdict = OrderedDict({  # Dimensional scaling:

                # Flow shape parameters
                'K_u': 0.5,
                'n_u': 0.5,

                # Small general diffusion factor:
                'Do': 1.0e-11,

                # Beta cat parameters
                'r_bc': 1.0e-3,
                'd_bc': 5.0e-7,
                'K_bc_apc': 0.5,
                'n_bc_apc': 1.0,
                'd_bc_deg': 3.0e-3,
                'K_bc_camp': 1.0,
                'n_bc_camp': 2.0,

                # ERK parameters
                'K_erk_bc': 10.0,
                'n_erk_bc': 2.0,

                # APC parameters
                'K_apc_wnt': 5.0,
                'n_apc_wnt': 2.0,

                # Hedgehog parameters:
                'r_hh': 5.0e-3,  # 2.5e-3
                'd_hh': 1.0e-5,
                'D_hh': 2.5e-11,
                'u_hh': 1.5e-7,

                # Wnt parameters
                'r_wnt': 5.0e-3,
                'd_wnt': 5.0e-6,
                'K_wnt_notum': 0.5,
                'n_wnt_notum': 2.0,
                'D_wnt': 0.75e-11,
                'd_wnt_deg_notum': 5.0e-3,
                'd_wnt_deg_ptc': 2.5e-5,
                'K_wnt_hh': 62.5,
                'n_wnt_hh': 2.0,
                'K_wnt_camp': 0.5,
                'n_wnt_camp': 2.0,

                # NRF parameters
                'r_nrf': 2.5e-3,
                'd_nrf': 1.0e-5,
                'K_nrf_bc': 100.0,
                'n_nrf_bc': 1.0,
                'D_nrf': 2.5e-11,
                'u_nrf': -1.5e-7,

                # Notum parameters
                'K_notum_nrf': 300.0,
                'n_notum_nrf': 2.5,
                'D_notum': 2.5e-11,

                # Markov model parameters:
                'C1': 0.65, # ERK constant to modulate head formation
                'K1': 0.025,

                'C2': 75.0, # Beta-catenin concentration to modulate tail formation
                'K2': 4.0,
                'Beta_HB': 1.0e-3, # head tissue decay time constant
                'Beta_TB': 1.0e-3, # tail tissue decay time constant

                'max_remod': 1.0e-2,  # maximum rate at which tissue remodelling occurs
                'hdac_decay': 5.0e-6,  # decay constant for hdac remodeling molecule
                'D_hdac': 1.0e-13, # diffusion constant for hdac remodeling molecule
                'hdac_o': 3.0  # initial concentration of hdac remodeling molecule


            })

        else:
            self.pdict = pdict

        # BETSE parameters object:
        self.p = Parameters.make(config_filename)

        self.verbose = verbose

        self.x_scale = xscale

        self.make_mesh()

        self.prime_model()

        # Initialize the transport field and nerve density:
        self.load_transport_field()

        self.runtype = 'init'

        if self.verbose is True:
            print("Successfully generated 1D model!")

    def make_mesh(self):

        if self.verbose is True:
            print("Creating 1D mesh...")

        # X-axis
        self.xmin = 0.0
        self.xmax = 0.01 * self.x_scale
        self.xmid = (self.xmax - self.xmin) / 2
        self.dx = 1.0e-4 * self.x_scale
        self.cdl = int((self.xmax - self.xmin) / self.dx)
        self.mdl = self.cdl - 1
        self.X = np.linspace(self.xmin, self.xmax, self.cdl)

        # Cut points -- indicies of the x-axis to cut along:
        self.cut_points = np.array([0.002, 0.004, 0.006, 0.008]) * self.x_scale
        self.cut_inds, _ = self.find_nearest(self.X, self.cut_points)
        self.get_seg_inds()

        # get indices surrounding each cut line:
        target_inds_wound = set()
        for i, j in self.seg_inds:
            if i > 0:
                target_inds_wound.add(i)
                target_inds_wound.add(i - 1)
                target_inds_wound.add(i + 1)

            if j < self.cdl:
                target_inds_wound.add(j)
                target_inds_wound.add(j - 1)
                target_inds_wound.add(j + 1)

        self.target_inds_wound = np.asarray(list(target_inds_wound))

        # build matrices
        self.build_matrices()

    def prime_model(self):

        if self.verbose is True:
            print("Initializing parameters and variables...")

        # tags for easy reference to concentrations of the model:
        self.conc_tags = ['β-Cat', 'Erk', 'Wnt', 'Hh', 'NRF', 'Notum', 'APC', 'cAMP']

        # Initialize the master molecules handlers to null values to avoid plot/animation issues:
        self.molecules_time = OrderedDict()
        self.molecules_time2 = OrderedDict()
        self.molecules_sim_time = OrderedDict()

        for tag in self.conc_tags:
            self.molecules_time[tag] = np.zeros(self.cdl)
            self.molecules_time2[tag] = np.zeros(self.cdl)
            self.molecules_sim_time[tag] = np.zeros(self.cdl)

        # Default RNAi keys:
        self.RNAi_defaults = {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1,
         'wnt': 1, 'hh': 1, 'camp': 1,'dynein': 1, 'kinesin':1}

        self.init_plots()

        # Beta cat parameters
        self.r_bc = self.pdict['r_bc']
        self.d_bc = self.pdict['d_bc']
        self.d_bc_deg = self.pdict['d_bc_deg']
        self.K_bc_apc = self.pdict['K_bc_apc']
        self.n_bc_apc = self.pdict['n_bc_apc']
        self.K_bc_camp = self.pdict['K_bc_camp']
        self.n_bc_camp = self.pdict['n_bc_camp']

        self.c_BC = np.ones(self.cdl)
        self.c_BC_time = []

        # ERK parameters
        self.r_erk = 5.0e-3
        self.d_erk = 5.0e-3
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.n_erk_bc = self.pdict['n_erk_bc']

        self.c_ERK = np.zeros(self.cdl)
        self.c_ERK_time = []

        # APC parameters
        self.r_apc = 5.0e-3
        self.d_apc = 5.0e-3
        self.K_apc_wnt = self.pdict['K_apc_wnt']
        self.n_apc_wnt = self.pdict['n_apc_wnt']

        self.c_APC = np.zeros(self.cdl)
        self.c_APC_time = []

        # Hedgehog parameters:
        self.r_hh = self.pdict['r_hh']
        self.d_hh = self.pdict['d_hh']
        self.D_hh = self.pdict['D_hh']
        self.u_hh = self.pdict['u_hh']

        self.c_HH = np.zeros(self.cdl)
        self.c_HH_time = []

        # Wnt parameters
        self.r_wnt = self.pdict['r_wnt']
        self.d_wnt = self.pdict['d_wnt']
        self.d_wnt_deg_notum = self.pdict['d_wnt_deg_notum']
        self.d_wnt_deg_ptc = self.pdict['d_wnt_deg_ptc']
        self.D_wnt = self.pdict['D_wnt']
        self.K_wnt_notum = self.pdict['K_wnt_notum']
        self.n_wnt_notum = self.pdict['n_wnt_notum']
        self.K_wnt_hh = self.pdict['K_wnt_hh']
        self.n_wnt_hh = self.pdict['n_wnt_hh']
        self.K_wnt_camp = self.pdict['K_wnt_camp']
        self.n_wnt_camp = self.pdict['n_wnt_camp']

        self.c_WNT = np.zeros(self.cdl)
        self.c_WNT_time = []

        # Notum regulating factor (NRF) parameters
        self.r_nrf = self.pdict['r_nrf']
        self.d_nrf = self.pdict['d_nrf']
        self.K_nrf_bc = self.pdict['K_nrf_bc']
        self.n_nrf_bc = self.pdict['n_nrf_bc']
        self.D_nrf = self.pdict['D_nrf']
        self.u_nrf = self.pdict['u_nrf']

        self.c_NRF = np.zeros(self.cdl)
        self.c_NRF_time = []

        # Notum parameters
        self.D_notum = self.pdict['D_notum']
        self.r_notum = 5.0e-3
        self.d_notum = 5.0e-3
        self.K_notum_nrf = self.pdict['K_notum_nrf']
        self.n_notum_nrf = self.pdict['n_notum_nrf']

        self.c_Notum = np.zeros(self.cdl)
        self.c_Notum_time = []

        # cAMP parameters
        self.r_camp = 5.0e-3
        self.d_camp = 5.0e-3

        self.c_cAMP = np.ones(self.cdl) * 1.0

        self.Do = self.pdict['Do']

        # Short x-axis:
        self.Xs = np.dot(self.Mx, self.X)

        # Markov model parameters:
        self.C1 = self.pdict['C1'] # ERK constant to modulate head formation
        self.K1 = self.pdict['K1']
        self.C2 = self.pdict['C2']  # Beta-catenin concentration to modulate tail formation
        self.K2 = self.pdict['K2']
        self.beta_HB = self.pdict['Beta_HB']  # head tissue decay time constant
        self.beta_TB = self.pdict['Beta_TB']  # tail tissue decay time constant
        self.alpha_BH = 1/(1 + np.exp(-(self.c_ERK - self.C1)/self.K1)) # init transition constant blastema to head
        self.alpha_BT = 1/(1 + np.exp(-(self.c_BC - self.C2)/self.K2)) # init transition constant blastema to tail

        self.max_remod = self.pdict['max_remod']  # maximum rate of tissue remodelling

        # initialize Markov model probabilities:
        self.Head = np.zeros(self.cdl) # head
        self.Tail = np.zeros(self.cdl) # tail
        self.Blast = np.ones(self.cdl) # blastema or stem cells

        # initialize remodeling molecule concentration:
        self.hdac =  np.zeros(self.cdl)
        self.hdac_decay = self.pdict['hdac_decay']
        self.D_hdac = self.pdict['D_hdac']
        self.hdac_o = self.pdict['hdac_o']

    def run_markov(self):
        """
        Updates the Markov model in time

        :return:

        """

        # update the remodelling-allowance molecule, hdac:
        # gradient and midpoint mean concentration:
        g_hdac, _ = self.get_gradient(self.hdac, self.runtype)

        flux = -g_hdac*self.D_hdac
        # divergence of the flux
        div_flux = self.get_div(flux, self.runtype)

        self.hdac += (-div_flux -self.hdac_decay*self.hdac)*self.dt

        # update transition constants based on new value of ERK and beta-Cat:
        self.alpha_BH = 1/(1 + np.exp(-(self.c_ERK - self.C1)/self.K1)) # init transition constant blastema to head
        self.alpha_BT = 1/(1 + np.exp(-(self.c_BC - self.C2)/self.K2)) # init transition constant blastema to tail

        delta_H = self.alpha_BH - self.Tail*self.alpha_BH - self.Head*(self.beta_HB + self.alpha_BH)
        delta_T = self.alpha_BT - self.Head*self.alpha_BT - self.Tail*(self.beta_TB + self.alpha_BT)

        # Update probabilities in time:
        self.Head += delta_H*self.dt*self.max_remod*self.hdac
        self.Tail += delta_T*self.dt*self.max_remod*self.hdac

        self.Blast = 1.0 - self.Head - self.Tail

    def load_transport_field(self):

        # Transport fields
        self.K_u = self.pdict['K_u']
        self.n_u = self.pdict['n_u']

        self.u = 1 / (1 + (self.Xs / (self.xmid / self.K_u)) ** self.n_u)

        self.NerveDensity = 1 / (1 + (self.X / (self.xmid / self.K_u)) ** self.n_u)

    def log_details(self):

        print("Spatial points: ", self.cdl)
        print("Time points: ", self.nt)
        print("Sampled time points: ", len(self.tsample))

    def build_matrices(self):
        """
        Builds gradient and laplacian matrices for whole model
        """

        Gx = np.zeros((self.cdl - 1, self.cdl))  # gradient matrix
        Mx = np.zeros((self.cdl - 1, self.cdl))  # averaging matrix

        for i, pi in enumerate(self.X):

            if i != self.cdl - 1:
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

                seg_inds.append([self.cut_inds[i], self.cdl])

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

    # GRN Updating functions---------------------------------------

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

        # change of bc:  #FIXME: BC produced on nerves or everywhere??
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

        term_hh = 1 / (1 + iHH)
        term_notum = iNotum / (1 + iNotum)
        term_camp = icAMP / (1 + icAMP)

        # Gradient and mean of concentration
        g_wnt, m_wnt = self.get_gradient(self.c_WNT, self.runtype)

        # Motor transport term:
        flux = -self.D_wnt * g_wnt

        # divergence
        div_flux = self.get_div(flux, self.runtype)

        del_wnt = (-div_flux + rnai * self.r_wnt * term_camp * self.NerveDensity -
                   self.d_wnt * self.c_WNT - self.d_wnt_deg_notum * term_notum * self.c_WNT
                                           - self.d_wnt_deg_ptc*term_hh*self.c_WNT)

        return del_wnt  # change in Wnt

    def update_hh(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in Hedgehog levels in space and time.
        """

        # Gradient and mean of concentration
        g_hh, m_hh = self.get_gradient(self.c_HH, self.runtype)

        # Motor transport term:
        conv_term = m_hh*self.u*self.u_hh*kinesin

        flux = -g_hh*self.D_hh + conv_term
        # flux = -g_hh * self.D_hh

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

    # GRN Running functions---------------------------------------

    def clear_cache_init(self):

        self.c_BC_time = []
        self.c_ERK_time = []
        self.c_WNT_time = []
        self.c_HH_time = []
        self.c_NRF_time = []
        self.c_Notum_time = []
        self.c_APC_time = []
        self.c_cAMP_time = []

        self.Head_time = []
        self.Tail_time = []
        self.Blast_time = []

        self.hdac_time = []

        self.delta_ERK_time = []

        # initialize remodeling molecule concentration to entire model:
        self.hdac =  self.hdac_o*np.ones(self.cdl)

    def clear_cache_reinit(self):

        self.c_BC_time2 = []
        self.c_ERK_time2 = []
        self.c_WNT_time2 = []
        self.c_HH_time2 = []
        self.c_NRF_time2 = []
        self.c_Notum_time2 = []
        self.c_APC_time2 = []
        self.c_cAMP_time2 = []

        self.hdac_time2 = []

        self.Head_time2 = []
        self.Tail_time2 = []
        self.Blast_time2 = []

        self.delta_ERK_time2 = []

    def clear_cache_sim(self):

        self.c_BC_sim_time = []
        self.c_ERK_sim_time = []
        self.c_WNT_sim_time = []
        self.c_HH_sim_time = []
        self.c_NRF_sim_time = []
        self.c_Notum_sim_time = []
        self.c_APC_sim_time = []
        self.c_cAMP_sim_time = []

        self.hdac_sim_time = []

        self.Head_sim_time = []
        self.Tail_sim_time = []
        self.Blast_sim_time = []

        self.delta_ERK_sim_time = []

        # initialize remodeling molecule concentration to wound edges only:
        self.hdac =  np.zeros(self.cdl)
        self.hdac[self.target_inds_wound] = self.hdac_o

    def run_loop(self,
                 knockdown=None):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        for tt in self.time:

            delta_bc = self.update_bc(rnai=knockdown['bc']) * self.dt  # time update beta-catenin
            delta_wnt = self.update_wnt(rnai=knockdown['wnt']) * self.dt  # time update wnt
            delta_hh = self.update_hh(rnai=knockdown['hh'],
                                      kinesin=knockdown['kinesin']) * self.dt  # time update hh
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

            # update the Markov model:
            self.run_markov()

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

                    self.hdac_time.append(self.hdac*1)

                    self.Head_time.append(self.Head*1)
                    self.Tail_time.append(self.Tail*1)
                    self.Blast_time.append(self.Blast*1)

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

                    self.Head_time2.append(self.Head*1)
                    self.Tail_time2.append(self.Tail*1)
                    self.Blast_time2.append(self.Blast*1)

                    self.hdac_time2.append(self.hdac * 1)

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

                    self.hdac_sim_time.append(self.hdac * 1)

                    self.Head_sim_time.append(self.Head*1)
                    self.Tail_sim_time.append(self.Tail*1)
                    self.Blast_sim_time.append(self.Blast*1)

                    self.delta_ERK_sim_time.append(delta_erk.mean() * 1)

    def initialize(self,
                   knockdown= None,
                   run_time=48.0 * 3600,
                   run_time_step=10,
                   run_time_sample=100,
                   reset_clims = True):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step*self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_init()
        self.runtype = 'init'
        self.run_loop(knockdown=knockdown)
        self.tsample_init = self.tsample

        # write the final molecule time arrays to the master dictionary:
        self.molecules_time['β-Cat'] = self.c_BC_time
        self.molecules_time['Erk'] = self.c_ERK_time
        self.molecules_time['Wnt'] = self.c_WNT_time
        self.molecules_time['Hh'] = self.c_HH_time
        self.molecules_time['NRF'] = self.c_NRF_time
        self.molecules_time['Notum'] = self.c_Notum_time
        self.molecules_time['APC'] = self.c_APC_time
        self.molecules_time['cAMP'] = self.c_cAMP_time

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = [0, np.max(self.molecules_time['β-Cat'])]
            mol_clims['Erk'] = [0, 1.0]
            mol_clims['Wnt'] = [0, np.max(self.molecules_time['Wnt'])]
            mol_clims['Hh'] = [0, np.max(self.molecules_time['Hh'])]
            mol_clims['NRF'] = [0, np.max(self.molecules_time['NRF'])]
            mol_clims['Notum'] = [0, 1.0]
            mol_clims['APC'] = [0, 1.0]
            mol_clims['cAMP'] = [0, 1.0]

            self.default_clims = mol_clims



        if self.verbose:
            print("Successfully completed init of 1D model!")

    def reinitialize(self,
                     knockdown=None,
                     run_time=48.0 * 3600,
                     run_time_step=10,
                     run_time_sample=100):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_reinit()
        self.runtype = 'reinit'
        self.run_loop(knockdown=knockdown)
        self.tsample_reinit = self.tsample

        # FIXME we don't need BC_time2 (etc) and 3 molecules_time
        # FIXME we should name molecules_reinit_time, molecules_init_time, etc

        # write the final molecule time arrays to the master dictionary:
        self.molecules_time2['β-Cat'] = self.c_BC_time2
        self.molecules_time2['Erk'] = self.c_ERK_time2
        self.molecules_time2['Wnt'] = self.c_WNT_time2
        self.molecules_time2['Hh'] = self.c_HH_time2
        self.molecules_time2['NRF'] = self.c_NRF_time2
        self.molecules_time2['Notum'] = self.c_Notum_time2
        self.molecules_time2['APC'] = self.c_APC_time2
        self.molecules_time2['cAMP'] = self.c_cAMP_time2

        if self.verbose:
            print("Successfully completed reinit of 1D model!")

    def simulate(self,
                 knockdown=None,
                 run_time=48.0 * 3600,
                 run_time_step=10,
                 run_time_sample=100,
                 reset_clims = False):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_sim()
        self.runtype = 'sim'
        self.run_loop(knockdown=knockdown)

        self.tsample_sim = self.tsample

        # write the final molecule time arrays to the master dictionary:
        self.molecules_sim_time['β-Cat'] = self.c_BC_sim_time
        self.molecules_sim_time['Erk'] = self.c_ERK_sim_time
        self.molecules_sim_time['Wnt'] = self.c_WNT_sim_time
        self.molecules_sim_time['Hh'] = self.c_HH_sim_time
        self.molecules_sim_time['NRF'] = self.c_NRF_sim_time
        self.molecules_sim_time['Notum'] = self.c_Notum_sim_time
        self.molecules_sim_time['APC'] = self.c_APC_sim_time
        self.molecules_sim_time['cAMP'] = self.c_cAMP_sim_time

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = [0, np.max(self.molecules_sim_time['β-Cat'])]
            mol_clims['Erk'] = [0, 1.0]
            mol_clims['Wnt'] = [0, np.max(self.molecules_sim_time['Wnt'])]
            mol_clims['Hh'] = [0, np.max(self.molecules_sim_time['Hh'])]
            mol_clims['NRF'] = [0, np.max(self.molecules_sim_time['NRF'])]
            mol_clims['Notum'] = [0, 1.0]
            mol_clims['APC'] = [0, 1.0]
            mol_clims['cAMP'] = [0, 1.0]

            self.default_clims = mol_clims

        if self.verbose:
            print("Successfully completed sim of 1D model!")

    # Plotting functions---------------------------------------

    def init_plots(self):

        # default plot legend scaling (can be modified)
        mol_clims = OrderedDict()

        mol_clims['β-Cat'] = [0, 100.0]
        mol_clims['Erk'] = [0, 1.0]
        mol_clims['Wnt'] = [0, 200.0]
        mol_clims['Hh'] = [0, 650.0]
        mol_clims['NRF'] = [0, 3000.0]
        mol_clims['Notum'] = [0, 1.0]
        mol_clims['APC'] = [0, 1.0]
        mol_clims['cAMP'] = [0, 1.0]

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


    def triplot(self, ti, plot_type = 'init',  fname = 'Triplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if cmaps is None:
            cmaps = self.default_cmaps

        if clims is None:
            clims = self.default_clims

        # Filesaving:
        if ti == -1:
            fstr = fname + '.png'

        else:
            fstr = fname + str(ti) + '.png'

        if dirsave is None and plot_type != 'sim':
            dirstr = os.path.join(self.p.init_export_dirname, 'Triplot')
        elif dirsave is None and plot_type == 'sim':
            dirstr = os.path.join(self.p.sim_export_dirname, 'Triplot')
        else:
            dirstr = dirsave

        fname = os.path.join(dirstr, fstr)

        os.makedirs(dirstr, exist_ok=True)

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]
            carray3 = self.molecules_sim_time['Notum'][ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)
            _, cs3 = self.get_plot_segs(carray3)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[2].plot(self.X * 1e3, carray3r, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data:
            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)
            axarr[2].plot(self.X*1e3, carray3, color=cmaps['Notum'], linewidth=linew)



        elif plot_type == 'sim':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)
                _, cs3r = self.get_plot_segs(carray3r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs3r):
                    axarr[2].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data
            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

            for xi, ci in zip(xs, cs3):
                axarr[2].plot(xi, ci, color=cmaps['Notum'], linewidth=linew)


        axarr[0].set_title("ERK")
        axarr[0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[0].set_ylim(clims['Erk'][0], clims['Erk'][1])

        axarr[1].set_title("beta-cat")
        axarr[1].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[1].set_ylim(clims['β-Cat'][0], clims['β-Cat'][1])

        axarr[2].set_title("Notum")
        axarr[2].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[2].set_ylim(clims['Notum'][0], clims['Notum'][1])

        axarr[2].set_xlabel('Axis Distance [mm]')

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0].transAxes)

        fig.subplots_adjust(hspace=0.15)
        fig.suptitle('Initialization', x=0.1, y=0.94)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string)

        plt.savefig(fname, format='png', dpi=reso)
        plt.close()

    def biplot(self, ti, plot_type = 'init', fname = 'Biplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
               ref_data=None, extra_text = None, txt_x = 0.05, txt_y = 0.92):


        if cmaps is None:
            cmaps = self.default_cmaps

        if clims is None:
            clims = self.default_clims

        # Filesaving:

        if ti == -1:
            fstr = fname + '.png'

        else:
            fstr = fname + str(ti) + '.png'

        if dirsave is None and plot_type != 'sim':
            dirstr = os.path.join(self.p.init_export_dirname, 'Triplot')
        elif dirsave is None and plot_type == 'sim':
            dirstr = os.path.join(self.p.sim_export_dirname, 'Triplot')
        else:
            dirstr = dirsave

        fname = os.path.join(dirstr, fstr)

        os.makedirs(dirstr, exist_ok=True)

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]


        elif plot_type == 'sim':
            tsample = self.tsample_sim

            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)


        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(2, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed')

            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)

        elif plot_type == 'sim':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

        axarr[0].set_title("ERK")
        axarr[0].set_ylabel('Concentration [nM]')

        if autoscale is False:
            axarr[0].set_ylim(clims['Erk'][0], clims['Erk'][1])

        axarr[1].set_title("beta-cat")
        axarr[1].set_ylabel('Concentration [nM]')

        if autoscale is False:
            axarr[1].set_ylim(clims['β-Cat'][0], clims['β-Cat'][1])

        axarr[1].set_xlabel('Axis Distance [mm]')

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0].transAxes)

        fig.subplots_adjust(hspace=0.15)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.1, y=0.94)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def plot(self, ti, ctag, plot_type='init', dirsave = 'Plot', reso = 150, linew = 3.0,
                cmaps=None, fontsize=16.0, fsize=(10, 6), clims = None, autoscale = True):

        if cmaps is None:
            cmaps = self.default_cmaps

        if clims is None:
            clims = self.default_clims

        # Plot an init:
        if plot_type == 'init':
            tsample = self.tsample_init
            carray = self.molecules_time[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':
            tsample = self.tsample_reinit
            carray = self.molecules_time2[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray = self.molecules_sim_time[ctag][ti]

            xs, cs1 = self.get_plot_segs(carray)

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.sim_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})
        plt.figure(figsize=fsize)
        ax = plt.subplot(111)

        if plot_type == 'init' or plot_type == 'reinit':

            plt.plot(self.X*1e3, carray, color=cmaps[ctag], linewidth=linew)

        elif plot_type == 'sim':

            for xi, ci in zip(xs, cs1):
                plt.plot(xi, ci, color=cmaps[ctag], linewidth=linew)

        if autoscale is False:
            ax.set_ylim(clims[ctag][0], clims[ctag][1])

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = 'Time: ' + str(tdays) + ' Hours'
        plt.title(tit_string)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def get_plot_segs(self, conc):

        xoo = self.X * 1e3

        sim_coo = []
        sim_xoo = []

        for a, b in self.seg_inds:
            sim_coo.append(conc[a:b])
            sim_xoo.append(xoo[a:b])

        sim_coo = np.asarray(sim_coo)
        sim_xoo = np.asarray(sim_xoo)

        return sim_xoo, sim_coo

    def animate_triplot(self, ani_type = 'init', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                        ref_data=None, extra_text = None):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.triplot(ii, plot_type='init', dirsave=dirsave, reso=reso, linew=linew,
                              cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale,
                             ref_data=ref_data, extra_text = extra_text)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.triplot(ii, plot_type='sim', dirsave=dirsave, reso=reso, linew=linew,
                              cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale,
                             ref_data=ref_data, extra_text = extra_text)


    def animate_biplot(self, ani_type='init', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                       ref_data=None, extra_text = None):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.biplot(ii, plot_type='init', dirsave=dirsave, reso=reso, linew=linew,
                            cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale,
                            ref_data=ref_data, extra_text=extra_text)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.biplot(ii, plot_type='sim', dirsave=dirsave, reso=reso, linew=linew,
                            cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale,
                            ref_data=ref_data, extra_text=extra_text)

    def animate_plot(self, ctag, ani_type='init', dirsave = 'PlotAni', reso = 150, linew = 3.0,
                cmaps=None, fontsize=16.0, fsize=(10, 6), clims = None, autoscale = True):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.plot(ii, ctag, plot_type='init', dirsave=dirsave, reso=reso, linew=linew,
                            cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):

                self.plot(ii, ctag, plot_type='sim', dirsave=dirsave, reso=reso, linew=linew,
                            cmaps=cmaps, fontsize=fontsize, fsize=fsize, autoscale=autoscale)

    def hexplot(self, ti, plot_type = 'init',  fname = 'Hexplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (16, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if cmaps is None:
            cmaps = self.default_cmaps

        if clims is None:
            clims = self.default_clims

        # Filesaving:
        if ti == -1:
            fstr = fname + '.png'

        else:
            fstr = fname + str(ti) + '.png'

        if dirsave is None and plot_type != 'sim':
            dirstr = os.path.join(self.p.init_export_dirname, 'Hexplot')
        elif dirsave is None and plot_type == 'sim':
            dirstr = os.path.join(self.p.sim_export_dirname, 'Hexplot')
        else:
            dirstr = dirsave

        fname = os.path.join(dirstr, fstr)

        os.makedirs(dirstr, exist_ok=True)

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]
            carray4 = self.molecules_time['Hh'][ti]
            carray5 = self.molecules_time['Wnt'][ti]
            carray6 = self.molecules_time['NRF'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]
            carray4 = self.molecules_time2['Hh'][ti]
            carray5 = self.molecules_time2['Wnt'][ti]
            carray6 = self.molecules_time2['NRF'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]
            carray3 = self.molecules_sim_time['Notum'][ti]
            carray4 = self.molecules_sim_time['Hh'][ti]
            carray5 = self.molecules_sim_time['Wnt'][ti]
            carray6 = self.molecules_sim_time['NRF'][ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)
            _, cs3 = self.get_plot_segs(carray3)
            _, cs4 = self.get_plot_segs(carray4)
            _, cs5 = self.get_plot_segs(carray5)
            _, cs6 = self.get_plot_segs(carray6)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, 2, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]
                carray4r = ref_data['Hh'][ti]
                carray5r = ref_data['Wnt'][ti]
                carray6r = ref_data['NRF'][ti]

                axarr[0,0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1,0].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[2,0].plot(self.X * 1e3, carray3r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[0, 1].plot(self.X * 1e3, carray4r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1, 1].plot(self.X * 1e3, carray5r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[2, 1].plot(self.X * 1e3, carray6r, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data:
            axarr[0,0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1,0].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)
            axarr[2,0].plot(self.X*1e3, carray3, color=cmaps['Notum'], linewidth=linew)
            axarr[0,1].plot(self.X*1e3, carray4, color=cmaps['Hh'], linewidth=linew)
            axarr[1,1].plot(self.X*1e3, carray5, color=cmaps['Wnt'], linewidth=linew)
            axarr[2,1].plot(self.X*1e3, carray6, color=cmaps['NRF'], linewidth=linew)



        elif plot_type == 'sim':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]
                carray4r = ref_data['Hh'][ti]
                carray5r = ref_data['Wnt'][ti]
                carray6r = ref_data['NRF'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)
                _, cs3r = self.get_plot_segs(carray3r)
                _, cs4r = self.get_plot_segs(carray4r)
                _, cs5r = self.get_plot_segs(carray5r)
                _, cs6r = self.get_plot_segs(carray6r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0,0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs2r):
                    axarr[1,0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs3r):
                    axarr[2,0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs4r):
                    axarr[0,1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs5r):
                    axarr[1,1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs6r):
                    axarr[2,1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data
            for xi, ci in zip(xs, cs1):
                axarr[0, 0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1, 0].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

            for xi, ci in zip(xs, cs3):
                axarr[2, 0].plot(xi, ci, color=cmaps['Notum'], linewidth=linew)

            for xi, ci in zip(xs, cs4):
                axarr[0, 1].plot(xi, ci, color=cmaps['Hh'], linewidth=linew)

            for xi, ci in zip(xs, cs5):
                axarr[1, 1].plot(xi, ci, color=cmaps['Wnt'], linewidth=linew)

            for xi, ci in zip(xs, cs6):
                axarr[2, 1].plot(xi, ci, color=cmaps['NRF'], linewidth=linew)



        axarr[0, 0].set_title("ERK")
        axarr[0, 0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[0,0].set_ylim(clims['Erk'][0], clims['Erk'][1])

        axarr[1,0].set_title("beta-cat")
        axarr[1,0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[1,0].set_ylim(clims['β-Cat'][0], clims['β-Cat'][1])

        axarr[2,0].set_title("Notum")
        axarr[2,0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[2,0].set_ylim(clims['Notum'][0], clims['Notum'][1])

        axarr[2,0].set_xlabel('Axis Distance [mm]')

        axarr[0, 1].set_title("Hh")
        axarr[0, 1].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[0,1].set_ylim(clims['Hh'][0], clims['Hh'][1])

        axarr[1,1].set_title("Wnt")
        axarr[1,1].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[1,1].set_ylim(clims['Wnt'][0], clims['Wnt'][1])

        axarr[2,1].set_title("NRF")
        axarr[2,1].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[2,1].set_ylim(clims['NRF'][0], clims['NRF'][1])

        axarr[2,1].set_xlabel('Axis Distance [mm]')

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0,0].transAxes)

        # fig.subplots_adjust(hspace=0.15)
        fig.suptitle('Initialization', x=0.1, y=0.94)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string)

        plt.savefig(fname, format='png', dpi=reso)
        plt.close()


    def markovplot(self, ti, plot_type = 'init',  fname = 'Markov_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if cmaps is None:
            cmaps = self.default_cmaps

        # Filesaving:
        if ti == -1:
            fstr = fname + '.png'

        else:
            fstr = fname + str(ti) + '.png'

        if dirsave is None and plot_type != 'sim':
            dirstr = os.path.join(self.p.init_export_dirname, 'MarkovPlot')
        elif dirsave is None and plot_type == 'sim':
            dirstr = os.path.join(self.p.sim_export_dirname, 'MarkovPlot')
        else:
            dirstr = dirsave

        fname = os.path.join(dirstr, fstr)

        os.makedirs(dirstr, exist_ok=True)

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            carray1 = self.Head_time[ti]
            carray2 = self.Tail_time[ti]
            carray3 = self.Blast_time[ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.Head_time2[ti]
            carray2 = self.Tail_time2[ti]
            carray3 = self.Blast_time2[ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.Head_sim_time[ti]
            carray2 = self.Tail_sim_time[ti]
            carray3 = self.Blast_sim_time[ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)
            _, cs3 = self.get_plot_segs(carray3)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[2].plot(self.X * 1e3, carray3r, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data:
            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)
            axarr[2].plot(self.X*1e3, carray3, color=cmaps['Notum'], linewidth=linew)



        elif plot_type == 'sim':

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]
                carray3r = ref_data['Notum'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)
                _, cs3r = self.get_plot_segs(carray3r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs3r):
                    axarr[2].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

            # main plot data
            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

            for xi, ci in zip(xs, cs3):
                axarr[2].plot(xi, ci, color=cmaps['Notum'], linewidth=linew)


        axarr[0].set_title("pHead")
        axarr[0].set_ylabel('Probability')
        if autoscale is False:
            axarr[0].set_ylim(0, 1)

        axarr[1].set_title("pTail")
        axarr[1].set_ylabel('Probability')
        if autoscale is False:
            axarr[1].set_ylim(0, 1)

        axarr[2].set_title("pBlastema")
        axarr[2].set_ylabel('Probability')
        if autoscale is False:
            axarr[2].set_ylim(0, 1)

        axarr[2].set_xlabel('Axis Distance [mm]')

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0].transAxes)

        fig.subplots_adjust(hspace=0.15)
        fig.suptitle('Initialization', x=0.1, y=0.94)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string)

        plt.savefig(fname, format='png', dpi=reso)
        plt.close()











