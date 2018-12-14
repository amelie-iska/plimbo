#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
2D Simulator object for Planarian Interface for PLIMBO module.
'''

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from matplotlib import rcParams

from collections import OrderedDict
import copy
import pickle
import os
import os.path
import sys, time
import csv

from scipy.ndimage import rotate
from scipy.misc import imresize

from betse.lib.pickle import pickles
from betse.science.simrunner import SimRunner
from betse.science.parameters import Parameters
from betse.science import filehandling as fh
from betse.science.math import modulate
from betse.util.path import files
from betse.science.phase.phasecls import SimPhase
from betse.science.phase.phaseenum import SimPhaseKind
from betse.science.math import toolbox as tb
from betse.util.type.mapping.mapcls import DynamicValue, DynamicValueDict


class PlanariaGRN2D(object):
    """
    Object describing 2D version of core GRN model.
    """

    def __init__(self, config_filename, pdict, xscale=1.0, new_mesh=False,
                 verbose=False):

        self.verbose = verbose

        # BETSE parameters object:
        self.p = Parameters.make(config_filename)

        # Factor scaling the size of the model:
        self.x_scale = xscale

        # Load or create 2D Voronoi mesh:
        self.make_mesh(new_mesh)

        # Save dictionary of parameters for the GRN model:
        self.pdict = pdict

        # Initialize parameters and concentration arrays:
        self.prime_model()

        # Initialize the transport field and nerve density:
        self.load_transport_field()

        # Flag to mark cutting event:
        self.model_has_been_cut = False

        if self.verbose is True:
            print("-----------------------------")
            print("Successfully generated 2D model!")
            print("-----------------------------")

    def prime_model(self):
        """
        Assign core parameters to model and initialize concentrations.

        """
        if self.verbose is True:
            print("Initializing parameters and variables.")

        pdict = self.pdict

        # tags for easy reference to concentrations of the model:
        self.conc_tags = ['β-Cat', 'Erk', 'Wnt', 'Hh', 'NRF', 'Notum', 'APC', 'cAMP']

        # Default RNAi keys:
        self.RNAi_defaults = {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1,
         'wnt': 1, 'hh': 1, 'camp': 1,'dynein': 1}

        self.init_plots()

        # Beta cat parameters
        self.r_bc = pdict['r_bc']
        self.d_bc = pdict['d_bc']
        self.d_bc_deg = pdict['d_bc_deg']
        self.K_bc_apc = pdict['K_bc_apc']
        self.n_bc_apc = pdict['n_bc_apc']
        self.K_bc_camp = pdict['K_bc_camp']
        self.n_bc_camp = pdict['n_bc_camp']

        self.c_BC = np.ones(self.cdl)

        # ERK parameters
        self.r_erk = 5.0e-3
        self.d_erk = 5.0e-3
        self.K_erk_bc = pdict['K_erk_bc']
        self.K_erk_bc = pdict['K_erk_bc']
        self.n_erk_bc = pdict['n_erk_bc']

        self.c_ERK = np.zeros(self.cdl)
        self.c_ERK_time = []

        # APC parameters
        self.r_apc = 5.0e-3
        self.d_apc = 5.0e-3
        self.K_apc_wnt = pdict['K_apc_wnt']
        self.n_apc_wnt = pdict['n_apc_wnt']

        self.c_APC = np.zeros(self.cdl)
        self.c_APC_time = []

        # Hedgehog parameters:
        self.r_hh = pdict['r_hh']
        self.d_hh = pdict['d_hh']
        self.D_hh = pdict['D_hh']
        #         self.u_hh = pdict['u_hh']

        self.c_HH = np.zeros(self.cdl)
        self.c_HH_time = []

        # Wnt parameters
        self.r_wnt = pdict['r_wnt']
        self.d_wnt = pdict['d_wnt']
        self.d_wnt_deg = pdict['d_wnt_deg']
        self.D_wnt = pdict['D_wnt']
        self.K_wnt_notum = pdict['K_wnt_notum']
        self.n_wnt_notum = pdict['n_wnt_notum']
        self.K_wnt_hh = pdict['K_wnt_hh']
        self.n_wnt_hh = pdict['n_wnt_hh']
        self.K_wnt_camp = pdict['K_wnt_camp']
        self.n_wnt_camp = pdict['n_wnt_camp']

        self.c_WNT = np.zeros(self.cdl)
        self.c_WNT_time = []

        # Notum regulating factor (NRF) parameters
        self.r_nrf = pdict['r_nrf']
        self.d_nrf = pdict['d_nrf']
        self.K_nrf_bc = pdict['K_nrf_bc']
        self.n_nrf_bc = pdict['n_nrf_bc']
        self.D_nrf = pdict['D_nrf']
        self.u_nrf = pdict['u_nrf']

        self.c_NRF = np.zeros(self.cdl)
        self.c_NRF_time = []

        # Notum parameters
        self.D_notum = pdict['D_notum']
        self.r_notum = 5.0e-3
        self.d_notum = 5.0e-3
        self.K_notum_nrf = pdict['K_notum_nrf']
        self.n_notum_nrf = pdict['n_notum_nrf']

        self.c_Notum = np.zeros(self.cdl)
        self.c_Notum_time = []

        # cAMP parameters
        self.r_camp = 5.0e-3
        self.d_camp = 5.0e-3

        # cAMP parameters
        self.c_cAMP = np.ones(self.cdl) * 1.0

        self.Do = pdict['Do']  # default diffusion constant for small smoothing

        self.no = pdict['no']  # offset to nerve density map

    def load_transport_field(self):
        """
        Loads and processes neural transport fields u(x,y) and production density
        gradients G(x,y) from external bitmap images.

        """
        # Load in nerve density estimate:
        raw_nerves, _ = modulate.gradient_bitmap(self.cells.cell_i,
                                                 self.cells, self.p)

        raw_nerves += self.no  # Factor added as an offset

        self.NerveDensity = raw_nerves / raw_nerves.max()

        mean_nerve = self.cells.meanval(self.NerveDensity)

        # Load in raw transport field from image:
        ux, _ = modulate.gradient_bitmap(self.cells.mem_i,
                                         self.cells, self.p,
                                         bitmap_filename=self.p.mtube_init_x)

        uy, _ = modulate.gradient_bitmap(self.cells.mem_i,
                                         self.cells, self.p,
                                         bitmap_filename=self.p.mtube_init_y)

        # calculate the initial magnitude:
        u_mag = self.cells.mag(ux, uy) + 1.0e-15

        # Normalize:
        ux = (ux / u_mag) * mean_nerve
        uy = (uy / u_mag) * mean_nerve

        #         ux = (ux/u_mag)
        #         uy = (uy/u_mag)

        # make the field divergence-free with respect to individual
        # control cells of the mesh (produces a smoother field):
        uxi, uyi = self.cells.single_cell_div_free(ux, uy)

        # average to the midpoint between two cell membranes:
        self.ux = self.cells.meanval(uxi)
        self.uy = self.cells.meanval(uyi)

        #         Get the average transport field at the cell centres:
        self.ucx, self.ucy = self.cells.average_vector(self.ux, self.uy)

        # cell-centre average magnitude
        self.uc_mag = self.cells.mag(self.ucx, self.ucy)

        # Final magnitude
        self.u_mag = self.cells.mag(self.ux, self.uy)

        self.runtype = 'init'

    def make_mesh(self, new_mesh=False):
        """
        Loads a saved 2D Voronoi mesh for the simulation.

        """

        if new_mesh is True:

            if self.verbose is True:
                print("Creating a new 2D Grid.")
            # Create a new grid:
            simrun = SimRunner(self.p)
            phase = simrun.seed()
            self.cells = phase.cells

        else:

            if not files.is_file(self.p.seed_pickle_filename):
                if self.verbose is True:
                    print("File not found; Creating a new 2D Grid")
                # Make a new mesh
                simrun = SimRunner(self.p)
                phase = simrun.seed()
                self.cells = phase.cells

            else:
                if self.verbose is True:
                    print("Loading a 2D Grid from file.")
                # Load from previous creation:
                self.cells, _ = fh.loadWorld(self.p.seed_pickle_filename)

        # assign commonly used dimensional parameters to the modeling object:
        self.assign_easy_x(self.cells)

        # indices for cutting event:----------------------------------
        phase_kind = SimPhaseKind.INIT

        # Simulation phase.
        phase = SimPhase(
            kind=phase_kind,
            #     callbacks=self._callbacks,
            cells=self.cells,
            p=self.p,
        )

        # Initialize core simulation data structures.
        phase.sim.init_core(phase)
        phase.dyna.init_profiles(phase)

        for cut_profile_name in phase.p.event_cut_profile_names:
            #     # Object picking the cells removed by this cut profile.
            tissue_picker = phase.dyna.cut_name_to_profile[
                cut_profile_name].picker

            #     # One-dimensional Numpy arrays of the indices of all
            #     # cells and cell membranes to be removed.
            self.target_inds_cell, self.target_inds_mem = (
                tissue_picker.pick_cells_and_mems(
                    cells=self.cells, p=self.p))

        self.phase = phase

    def assign_easy_x(self, cells):

        # Short forms of commonly used mesh components:
        self.xc = cells.cell_centres[:, 0]
        self.yc = cells.cell_centres[:, 1]

        # Midpoints between cell centres:
        self.xm = cells.meanval(self.xc)
        self.ym = cells.meanval(self.yc)

        self.xmem = cells.mem_mids_flat[:, 0]
        self.ymem = cells.mem_mids_flat[:, 1]

        self.xec = cells.ecm_mids[:, 0]
        self.yec = cells.ecm_mids[:, 1]

        self.xenv = cells.xypts[:, 0]
        self.yenv = cells.xypts[:, 1]

        self.xyaxis = [cells.xmin, cells.xmax, cells.ymin, cells.ymax]

        self.verts = cells.cell_verts

        self.verts_r = self.rotate_verts(self.verts, -45.0)

        self.nx = cells.mem_vects_flat[:, 2]
        self.ny = cells.mem_vects_flat[:, 3]

        # assign length of cell (cdl) and mems (mdl) of mesh:
        self.cdl = len(cells.cell_i)
        self.mdl = len(cells.mem_i)

        # rotate x and y axes to the vertical
        self.xcr, self.ycr = self.rotate_field(-45.0, self.xc, self.yc)

    def rotate_verts(self, cellverts, angle_o, trans_x = 0.0, trans_y = 0.0):
        """
        Rotates mesh vertices by an angle_o (in degrees).
        """

        angle_i = (angle_o * np.pi) / 180.0
        csa = np.cos(angle_i)
        sna = np.sin(angle_i)

        cell_verts_2 = []

        for verts in cellverts:
            new_x = verts[:, 0] * csa - verts[:, 1] * sna + trans_x
            new_y = verts[:, 1] * csa + verts[:, 0] * sna + trans_y

            verti = np.column_stack((new_x, new_y))

            cell_verts_2.append(verti)

        cell_verts_2 = np.asarray(cell_verts_2)

        return cell_verts_2

    def rotate_field(self, angle_r, xc, yc):
        """
        Rotates components of an x, y field
        through angle_r (in degrees).
        """

        angle = (angle_r * np.pi) / 180.0
        cosa = np.cos(angle)
        sina = np.sin(angle)

        xcr = xc * cosa - yc * sina
        ycr = yc * cosa + xc * sina

        return xcr, ycr

    def cut_cells(self):
        """
        Removes cells from the Voronoi mesh by user-specified cutting profiles.

        """

        if self.model_has_been_cut is False:

            new_cell_centres = []
            new_ecm_verts = []
            removal_flags = np.zeros(len(self.cells.cell_i))
            removal_flags[self.target_inds_cell] = 1

            for i, flag in enumerate(removal_flags):
                if flag == 0:
                    new_cell_centres.append(self.cells.cell_centres[i])
                    new_ecm_verts.append(self.cells.ecm_verts[i])

            self.cells.cell_centres = np.asarray(new_cell_centres)
            self.cells.ecm_verts = np.asarray(new_ecm_verts)

            # recalculate ecm_verts_unique:
            ecm_verts_flat, _, _ = tb.flatten(self.cells.ecm_verts)
            ecm_verts_set = set()

            for vert in ecm_verts_flat:
                ptx = vert[0]
                pty = vert[1]
                ecm_verts_set.add((ptx, pty))

            self.cells.ecm_verts_unique = [list(verts) for verts in list(ecm_verts_set)]
            self.cells.ecm_verts_unique = np.asarray(self.cells.ecm_verts_unique)  # convert to numpy arra

            self.cells.cellVerts(self.p)  # create individual cell polygon vertices and other essential data structures
            self.cells.cellMatrices(self.p)  # creates a variety of matrices used in routine cells calculations
            self.cells.intra_updater(self.p)  # creates matrix used for finite volume integration on cell patch
            self.cells.cell_vols(self.p)  # calculate the volume of cell and its internal regions
            self.cells.mem_processing(self.p)  # calculates membrane nearest neighbours, ecm interaction, boundary tags, etc
            self.cells.near_neigh(self.p)  # Calculate the nn array for each cell
            self.cells.voronoiGrid(self.p)
            self.cells.calc_gj_vects(self.p)
            self.cells.environment(self.p)  # define features of the ecm grid
            self.cells.make_maskM(self.p)
            self.cells.grid_len = len(self.cells.xypts)

            #         self.cells.graphLaplacian(self.p)

            # re-do tissue profiles and GJ
            self.phase.dyna.init_profiles(self.phase)
            self.cells.redo_gj(self.phase)  # redo gap junctions to isolate different tissue types

            # reassign commonly used quantities
            self.assign_easy_x(self.cells)

            # assign updated cells object to a simulation object for plotting
            self.cells_s = copy.deepcopy(self.cells)

            # Cut model variable arrays to new dimenstions:
            self.c_BC = np.delete(self.c_BC, self.target_inds_cell)
            self.c_ERK = np.delete(self.c_ERK, self.target_inds_cell)
            self.c_HH = np.delete(self.c_HH, self.target_inds_cell)
            self.c_WNT = np.delete(self.c_WNT, self.target_inds_cell)
            self.c_Notum = np.delete(self.c_Notum, self.target_inds_cell)
            self.c_NRF = np.delete(self.c_NRF, self.target_inds_cell)
            self.c_APC = np.delete(self.c_APC, self.target_inds_cell)
            self.c_cAMP = np.delete(self.c_cAMP, self.target_inds_cell)

            self.NerveDensity = np.delete(self.NerveDensity, self.target_inds_cell)
            self.ux = np.delete(self.ux, self.target_inds_mem)
            self.uy = np.delete(self.uy, self.target_inds_mem)

            # magnitude of updated transport field
            self.u_mag = self.cells.mag(self.ux, self.uy)

            # Get the average transport field at the cell centres:
            self.ucx, self.ucy = self.cells.average_vector(self.ux, self.uy)
            # cell-centre average magnitude
            self.uc_mag = self.cells.mag(self.ucx, self.ucy)

            self.model_has_been_cut = True

    def scale_cells(self, xscale):
        """

        Scale the cell cluster object, and all related mathematical operators,
        by a factor 'xscale'

        """

        cells = copy.deepcopy(self.cells)

        self.p.cell_radius = self.p.cell_radius * xscale
        cells.cell_centres = self.cells.cell_centres * xscale

        new_ecm_verts = []

        for verti in self.cells.ecm_verts:
            vi = xscale * np.array(verti)
            new_ecm_verts.append(vi)

        cells.ecm_verts = np.asarray(new_ecm_verts)

        # recalculate ecm_verts_unique:
        ecm_verts_flat, _, _ = tb.flatten(cells.ecm_verts)
        ecm_verts_set = set()

        for vert in ecm_verts_flat:
            ptx = vert[0]
            pty = vert[1]
            ecm_verts_set.add((ptx, pty))

        cells.ecm_verts_unique = [list(verts) for verts in list(ecm_verts_set)]
        cells.ecm_verts_unique = np.asarray(cells.ecm_verts_unique)

        cells.cellVerts(self.p)  # create individual cell polygon vertices and other essential data structures
        cells.cellMatrices(self.p)  # creates a variety of matrices used in routine cells calculations
        cells.intra_updater(self.p)  # creates matrix used for finite volume integration on cell patch
        cells.cell_vols(self.p)  # calculate the volume of cell and its internal regions
        cells.mem_processing(self.p)  # calculates membrane nearest neighbours, ecm interaction, boundary tags, etc
        cells.near_neigh(self.p)  # Calculate the nn array for each cell
        cells.voronoiGrid(self.p)
        cells.calc_gj_vects(self.p)
        cells.environment(self.p)  # define features of the ecm grid
        cells.make_maskM(self.p)
        cells.grid_len = len(cells.xypts)

        #         cells.graphLaplacian(self.p)

        # reassign commonly used quantities
        self.assign_easy_x(cells)

        return cells

    # GRN Updating functions---------------------------------------

    def update_bc(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in beta-catenin levels in space and time.
        """

        # Growth and decay
        iAPC = (self.c_APC / self.K_bc_apc) ** self.n_bc_apc
        term_apc = iAPC / (1 + iAPC)

        icAMP = (self.c_cAMP / self.K_bc_camp) ** self.n_bc_camp
        term_camp = 1 / (1 + icAMP)

        # gradient of concentration:
        _, g_bcx, g_bcy = self.cells.gradient(self.c_BC)

        # flux:
        fx = -g_bcx * self.Do
        fy = -g_bcy * self.Do

        # divergence of the flux:
        div_flux = self.cells.div(fx, fy, cbound=True)

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

        # Gradient and midpoint mean of concentration
        _, g_nrf_x, g_nrf_y = self.cells.gradient(self.c_NRF)

        m_nrf = self.cells.meanval(self.c_NRF)

        # Motor transport term:
        conv_term_x = m_nrf * self.ux * self.u_nrf * dynein
        conv_term_y = m_nrf * self.uy * self.u_nrf * dynein

        fx = -g_nrf_x * self.D_nrf + conv_term_x
        fy = -g_nrf_y * self.D_nrf + conv_term_y

        div_flux = self.cells.div(fx, fy, cbound=True)

        # divergence of flux, growth and decay, breakdown in chemical tagging reaction:
        del_nrf = (-div_flux + self.r_nrf * term_bc - self.d_nrf * self.c_NRF)

        return del_nrf  # change in NRF

    def update_notum(self, rnai=1.0):
        """
        Method describing change in Notum levels in space and time.
        """

        iNRF = (self.c_NRF / self.K_notum_nrf) ** self.n_notum_nrf

        term_nrf = iNRF / (1 + iNRF)

        # Gradient of concentration
        _, g_not_x, g_not_y = self.cells.gradient(self.c_Notum)

        fx = -g_not_x * self.D_notum
        fy = -g_not_y * self.D_notum

        div_flux = self.cells.div(fx, fy, cbound=True)

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

        # Gradient of concentration
        _, g_wnt_x, g_wnt_y = self.cells.gradient(self.c_WNT)

        # Motor transport term:
        fx = -self.D_wnt * g_wnt_x
        fy = -self.D_wnt * g_wnt_y

        # divergence
        div_flux = self.cells.div(fx, fy, cbound=True)

        del_wnt = (-div_flux + rnai * self.r_wnt * term_hh * term_camp * self.NerveDensity -
                   self.d_wnt * self.c_WNT - self.d_wnt_deg * term_notum * self.c_WNT)

        return del_wnt  # change in Wnt

    def update_hh(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in Hedgehog levels in space and time.
        """

        # Gradient of concentration
        _, g_hh_x, g_hh_y = self.cells.gradient(self.c_HH)

        #         Motor transport term:
        #         m_hh = self.cells.meanval(self.c_HH)
        #         conv_term = m_hh*self.u*self.u_hh*kinesin
        #         flux = -g_hh*self.D_hh + conv_term

        fx = -g_hh_x * self.D_hh
        fy = -g_hh_y * self.D_hh

        #         divergence
        div_flux = self.cells.div(fx, fy, cbound=True)

        # final change in hh
        del_hh = (-div_flux + rnai * self.r_hh * self.NerveDensity - self.d_hh * self.c_HH)

        return del_hh  # change in Hedgehog

    def update_erk(self, rnai=1.0):
        """
        Method describing change in ERK levels in space and time.
        """

        iBC = (self.c_BC / self.K_erk_bc) ** self.n_erk_bc

        term_bc = 1 / (1 + iBC)

        _, g_erk_x, g_erk_y = self.cells.gradient(self.c_ERK)

        fx = -g_erk_x * self.Do
        fy = -g_erk_y * self.Do

        #         divergence
        div_flux = self.cells.div(fx, fy, cbound=True)

        del_erk = -div_flux + rnai * self.r_erk * term_bc - self.d_erk * self.c_ERK

        return del_erk

    def update_apc(self, rnai=1.0):
        """
        Method describing change in APC levels in space and time.
        """

        iWNT = (self.c_WNT / self.K_apc_wnt) ** self.n_apc_wnt

        term_wnt = 1 / (1 + iWNT)

        _, g_apc_x, g_apc_y = self.cells.gradient(self.c_APC)

        fx = -g_apc_x * self.Do
        fy = -g_apc_y * self.Do

        #         divergence
        div_flux = self.cells.div(fx, fy, cbound=True)

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

    def run_loop(self, knockdown = None):

        if knockdown is None:
            knockdown = self.RNAi_defaults

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

    def test_run(self, run_time=100.0,
                 run_time_step=10.0,
                 run_time_sample=100):

        # Test concentration to ensure conservatio of mass
        termx = (self.xc / self.xc.mean()) ** 2
        self.c_Test = termx / (1 + termx)

        self.c_Test_time = []

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = run_time_sample
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        Dtest = 5.0e-11
        utest = -1.0e-7
        rtest = 0.0
        dtest = 0.0

        for tt in self.time:

            # Gradient of concentration
            _, g_x, g_y = self.cells.gradient(self.c_Test)

            #         Motor transport term:
            m_test = self.cells.meanval(self.c_Test)

            conv_x = m_test * self.ux * utest
            conv_y = m_test * self.uy * utest

            fx = -g_x * Dtest + conv_x
            fy = -g_y * Dtest + conv_y

            #             fx = -g_x*Dtest
            #             fy = -g_y*Dtest

            #         divergence
            div_flux = self.cells.div(fx, fy, cbound=True)

            # final change in test
            del_test = (-div_flux + rtest - dtest * self.c_Test)

            self.c_Test += del_test * self.dt

            if tt in self.tsample:
                self.c_Test_time.append(self.c_Test * 1)

        self.tot_conc_time = np.asarray(
            [(ci * self.cells.cell_vol).sum() for ci in self.c_Test_time]
        )

    def initialize(self,
                   knockdown=None,
                   run_time=48.0 * 3600,
                   run_time_step=10,
                   run_time_sample=100,
                   reset_clims = True):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = run_time_sample
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        if self.x_scale != 1.0:
            # scale the cell cluster by the given factor
            self.cells = self.scale_cells(self.x_scale)

        # save the cells object at this stage so that we can plot init later
        self.cells_i = copy.deepcopy(self.cells)

        self.clear_cache_init()
        self.runtype = 'init'
        self.run_loop(knockdown=knockdown)
        self.tsample_init = self.tsample

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = (0, np.max(self.molecules_time['β-Cat']))
            mol_clims['Erk'] = (0, 1.0)
            mol_clims['Wnt'] = (0, np.max(self.molecules_time['Wnt']))
            mol_clims['Hh'] = (0, np.max(self.molecules_time['Hh']))
            mol_clims['NRF'] = (0, np.max(self.molecules_time['NRF']))
            mol_clims['Notum'] = (0, 1.0)
            mol_clims['APC'] = (0, 1.0)
            mol_clims['cAMP'] = (0, 1.0)

            self.default_clims = mol_clims

        if self.verbose:
            print("-----------------------------")
            print("Successfully completed init of 2D model!")
            print("-----------------------------")

    def reinitialize(self,
                     knockdown= None,
                     run_time=48.0 * 3600,
                     run_time_step=10,
                     run_time_sample=100):

        if knockdown is None:
            knockdown = self.RNAi_defaults

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
            print("Successfully completed reinit of 2D model!")
            print("-----------------------------")

    def simulate(self,
                 knockdown = None,
                 run_time=48.0 * 3600,
                 run_time_step=10,
                 run_time_sample=100,
                 reset_clims=False):

        if knockdown is None:
            knockdown = self.RNAi_defaults

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

        # Remove cells at user-specified cutlines and update all array indices:
        self.cut_cells()

        self.run_loop(knockdown=knockdown)

        self.tsample_sim = self.tsample

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = (0, np.max(self.molecules_sim_time['β-Cat']))
            mol_clims['Erk'] = (0, 1.0)
            mol_clims['Wnt'] = (0, np.max(self.molecules_sim_time['Wnt']))
            mol_clims['Hh'] = (0, np.max(self.molecules_sim_time['Hh']))
            mol_clims['NRF'] = (0, np.max(self.molecules_sim_time['NRF']))
            mol_clims['Notum'] = (0, 1.0)
            mol_clims['APC'] = (0, 1.0)
            mol_clims['cAMP'] = (0, 1.0)

            self.default_clims = mol_clims

        if self.verbose:
            print("-----------------------------")
            print("Successfully completed sim of 2D model!")
            print("-----------------------------")

    # Plotting functions---------------------------------------

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

        mol_cmaps['β-Cat'] = cm.magma
        mol_cmaps['Erk'] = cm.RdBu_r
        mol_cmaps['Wnt'] = cm.Spectral
        mol_cmaps['Hh'] = cm.Spectral
        mol_cmaps['NRF'] = cm.Spectral
        mol_cmaps['Notum'] = cm.PiYG_r
        mol_cmaps['APC'] = cm.Spectral
        mol_cmaps['cAMP'] = cm.Spectral

        self.default_cmaps = mol_cmaps

    def triplot(self, ti, plot_type='init', autoscale=True, dirsave='Triplot', reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 8), axisoff=False):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]

            fstr = 'Triplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]

            fstr = 'Triplot2_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=fsize)

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])

        col1 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Erk'], linewidth=0.0)
        if autoscale is False:
            col1.set_clim(clims['Erk'][0], clims['Erk'][1])
        col1.set_array(carray1)



        col2 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['β-Cat'], linewidth=0.0)
        if autoscale is False:
            col2.set_clim(clims['β-Cat'][0], clims['β-Cat'][1])
        col2.set_array(carray2)


        col3 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Notum'], linewidth=0.0)
        if autoscale is False:
            col3.set_clim(clims['Notum'][0], clims['Notum'][1])
        col3.set_array(carray3)


        ax1.add_collection(col1)

        ax1.set_title('Erk')
        ax1.axis('tight')
        ax1.axis('off')

        ax2.add_collection(col2)
        ax2.set_title('β-Cat')
        ax2.axis('tight')
        ax2.axis('off')

        ax3.add_collection(col3)
        ax3.set_title('Notum')
        ax3.axis('tight')
        ax3.axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        fig.subplots_adjust(wspace=0.0)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def biplot(self, ti, plot_type='init', autoscale=True, dirsave='Biplot', reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(10, 6), axisoff=False):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':

            tsample = self.tsample_init
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]

            fstr = 'Biplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]

            fstr = 'Biplot2_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]

            fstr = 'Biplot_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.sim_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=fsize)

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        col1 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Erk'], linewidth=0.0)
        col1.set_array(carray1)

        if autoscale is False:
            col1.set_clim(clims['Erk'])

        col2 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['β-Cat'], linewidth=0.0)
        col2.set_array(carray2)

        if autoscale is False:
            col2.set_clim(clims['β-Cat'])

        ax1.add_collection(col1)
        ax1.axis('tight')
        ax1.set_title('Erk')
        ax1.axis('off')

        ax2.add_collection(col2)
        ax2.axis('tight')
        ax2.set_title('β-Cat')
        ax2.axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        if axisoff is True:
            plt.axis('off')

        fig.subplots_adjust(wspace=0.0)

        plt.savefig(fname, format='png', dpi=reso, transparent = True)
        plt.close()

    def plot(self, ti, ctag, plot_type='init', autoscale=True, dirsave = 'Plot', reso = 150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(4, 6), axisoff = False):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps

        # Plot an init:
        if plot_type == 'init':
            tsample = self.tsample_init
            self.assign_easy_x(self.cells_i)
            carray = self.molecules_time[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'

            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'reinit':
            tsample = self.tsample_reinit
            self.assign_easy_x(self.cells_i)
            carray = self.molecules_time2[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.init_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            self.assign_easy_x(self.cells_s)
            carray = self.molecules_sim_time[ctag][ti]

            fstr = ctag + '_' + str(ti) + '_.png'
            dirstr = os.path.join(self.p.sim_export_dirname, dirsave)
            fname = os.path.join(dirstr, fstr)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})
        plt.figure(figsize=fsize)
        ax = plt.subplot(111)

        col1 = PolyCollection(self.verts_r * 1e3, edgecolor=None,
                              cmap=cmaps[ctag], linewidth=0.0)
        col1.set_array(carray)

        if autoscale is False:
            col1.set_clim(clims[ctag])

        ax.add_collection(col1)
        plt.colorbar(col1)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        plt.title(tit_string)

        plt.axis('equal')

        if axisoff is True:
            plt.axis('off')

        plt.savefig(fname, format='png', dpi=reso, transparent = True)
        plt.close()

    def animate_triplot(self, ti, ani_type='init', autoscale=True, dirsave='TriplotAni', reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 8), axisoff=False):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.triplot(ii, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.triplot(ii, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

    def animate_biplot(self, ti, ani_type='init', autoscale=True, dirsave='BiplotAni', reso=150,
               clims=None, cmaps=None, fontsize=18.0, fsize=(10, 6), axisoff=False):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.biplot(ii, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.biplot(ii, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

    def animate_plot(self, ti, ctag, ani_type='init', autoscale=True, dirsave='PlotAni', reso=150,
             clims=None, cmaps=None, fontsize=18.0, fsize=(4, 6), axisoff=False):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.plot(ii, ctag, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.plot(ii, ctag, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

