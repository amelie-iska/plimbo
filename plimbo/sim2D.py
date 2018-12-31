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
# from betse.util.type.mapping.mapcls import DynamicValue, DynamicValueDict
from plimbo.simabc import PlanariaGRNABC
from sklearn.cluster import DBSCAN


class PlanariaGRN2D(PlanariaGRNABC):
    """
    Object describing 2D version of core GRN model.
    """

    def __init__(self, *args, **kwargs):

        self.model_init(*args, **kwargs)

    def run_markov(self, ti):
        """
        Updates the Markov model in time

        :return:
        """

        # update the remodelling-allowance molecule, hdac:
        # gradient of concentration:
        _, g_hdacx, g_hdacy = self.cells.gradient(self.hdac)

        fx = -g_hdacx*self.D_hdac
        fy = -g_hdacy*self.D_hdac

        # divergence of the flux
        div_flux = self.cells.div(fx, fy, cbound=True)

        # growth characteristics for mode masked to certain areas, depending on run type:
        gmod = np.ones(self.cdl)
        gpulse = 1.0

        if self.runtype == 'sim': # limit growth to wounds for a timed process:
            gmod = np.zeros(self.cdl)
            gmod[self.target_inds_wound] = 1.0
            gpulse = 1.0 - tb.step(ti, self.hdac_to, self.hdac_ts)

        elif self.runtype == 'reinit':
            gpulse = 0.0 # inhibit grwoth of hdac

        self.hdac += (-div_flux + gmod*gpulse*self.hdac_growth -self.hdac_growth*self.hdac)*self.dt

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
        """
        Loads and processes neural transport fields u(x,y) and production density
        gradients G(x,y) from external bitmap images.

        """

        self.no = 0.5  # offset to nerve density map
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

        # ux = (ux/u_mag)
        # uy = (uy/u_mag)

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

            # get the corresponding flags to nearest neighbours
            target_inds_nn, _, _ = tb.flatten(self.cells.cell_to_nn_full[self.target_inds_cell])

            # create mask to mark wound edges of remaining cluster ------------------------------------
            target_inds_nn_unique = np.unique(target_inds_nn)

            hurt_cells = np.zeros(len(self.cells.cell_i))

            for i, inds in enumerate(self.cells.cell_to_nn_full):  # for all the nn inds to a cell...
                inds_array = np.asarray(inds)
                inds_in_target = np.intersect1d(inds_array, target_inds_nn_unique)

                if len(inds_in_target):
                    hurt_cells[i] = 1  # flag the cell as a "hurt" cell

            hurt_inds = (hurt_cells == 1).nonzero()
            self.hurt_mask = np.zeros(self.cdl)
            self.hurt_mask[hurt_inds] = 1.0

            #----------------------------------------------------------------------------------------------

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

            # Cut model variable arrays to new dimensions:
            self.c_BC = np.delete(self.c_BC, self.target_inds_cell)
            self.c_ERK = np.delete(self.c_ERK, self.target_inds_cell)
            self.c_HH = np.delete(self.c_HH, self.target_inds_cell)
            self.c_WNT = np.delete(self.c_WNT, self.target_inds_cell)
            self.c_Notum = np.delete(self.c_Notum, self.target_inds_cell)
            self.c_NRF = np.delete(self.c_NRF, self.target_inds_cell)
            self.c_APC = np.delete(self.c_APC, self.target_inds_cell)
            self.c_cAMP = np.delete(self.c_cAMP, self.target_inds_cell)

            self.Head = np.delete(self.Head, self.target_inds_cell)
            self.Tail = np.delete(self.Tail, self.target_inds_cell)
            self.Blast = np.delete(self.Blast, self.target_inds_cell)
            self.hdac = np.delete(self.hdac, self.target_inds_cell)

            self.NerveDensity = np.delete(self.NerveDensity, self.target_inds_cell)
            self.ux = np.delete(self.ux, self.target_inds_mem)
            self.uy = np.delete(self.uy, self.target_inds_mem)

            # remove cells from the "hurt mask"
            self.hurt_mask = np.delete(self.hurt_mask, self.target_inds_cell)

            # magnitude of updated transport field
            self.u_mag = self.cells.mag(self.ux, self.uy)

            # Get the average transport field at the cell centres:
            self.ucx, self.ucy = self.cells.average_vector(self.ux, self.uy)

            # cell-centre average magnitude
            self.uc_mag = self.cells.mag(self.ucx, self.ucy)

            # identify indices targets for wounded cells in the new model.
            match_inds = (self.hurt_mask == 1.0).nonzero()[0]
            target_inds_wound = match_inds

            # we want a few cell layers around the wound, so assign these:
            next_inds_wound = []

            for i, indy in enumerate(self.cells.cell_nn_i[:, 0]):

                if indy in target_inds_wound:
                    next_inds_wound.append(self.cells.cell_nn_i[i, 1])

            self.target_inds_wound = np.asarray(next_inds_wound)

            # update the hurt_mask to reflect thicker cell layers around wounds:
            self.hurt_mask[self.target_inds_wound] = 1

                    # identify clusters of indices representing each fragment:
            self.fragments, self.frag_xy, self.frag_xyr = self.cluster_points(self.cells.cell_i, dmax = 2.0)

            # idenify wounds within each fragment:
            self.wounds, self.wound_xy, self.wound_xyr = self.cluster_points(self.target_inds_wound, dmax=2.0)

            # Organize wounds into fragments:
            self.frags_and_wounds = OrderedDict()

            for fragi in self.fragments.keys():
                # self.frags_and_wounds[fragi] = OrderedDict()
                self.frags_and_wounds[fragi] = []

            for fragn, fragi in self.fragments.items():

                for woundn, woundi in self.wounds.items():

                    intersecto = np.intersect1d(woundi, fragi)

                    if len(intersecto):
                        # self.frags_and_wounds[fragn][woundn] = intersecto
                        self.frags_and_wounds[fragn].append(intersecto)

            self.model_has_been_cut = True

    def cluster_points(self, cinds, dmax = 2.0):
        """
        Identifies clusters of points (e.g. fragment or wounded zones within a fragment)

        cinds: indices to self.cells.cell_centers array, or a subset (typically self.cells.cell_i)
        dmax: multiplier of self.p.cell_radius (maximum nearest-neighbour distance)
\
        """

        all_cell_i = np.asarray(self.cells.cell_i)

        maxd = dmax * self.p.cell_radius # maximum search distance to nearest-neighbours

        cell_centres = self.cells.cell_centres[cinds] # get relevant cloud of x,y points to cluster

        cell_i = all_cell_i[cinds] # get relevant indices for the case we're working with a subset

        clust = DBSCAN(eps=maxd, min_samples=4).fit(cell_centres) # Use scikitlearn to flag clusters

        # organize the data:
        clusters = OrderedDict()  # dictionary holding indices of clusters
        cluster_xyr = OrderedDict()  # dictionary of rotated cluster x,y point centers
        cluster_xy = OrderedDict()  # dictionary of cluster x,y point centers

        for li in clust.labels_:
            clusters[li] = []

        for ci, li in zip(cell_i, clust.labels_):
            clusters[li].append(ci)

        for clustn, clusti in clusters.items():
            pts = self.cells.cell_centres[clusti]

            ptx = self.xcr[clusti]
            pty = self.ycr[clusti]

            cluster_xyr[clustn] = (ptx.mean(), pty.mean())
            cluster_xy[clustn] = (pts[0].mean(), pts[1].mean())

        return clusters, cluster_xy, cluster_xyr

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
        m_bc = self.cells.meanval(self.c_BC)

        # Motor transport term:
        conv_term_x = m_bc * self.ux * self.u_bc * kinesin
        conv_term_y = m_bc * self.uy * self.u_bc * kinesin

        # flux:
        fx = -g_bcx * self.Do + conv_term_x
        fy = -g_bcy * self.Do + conv_term_y

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

        term_hh = 1 / (1 + iHH)
        term_notum = iNotum / (1 + iNotum)
        term_camp = icAMP / (1 + icAMP)

        # Gradient of concentration
        _, g_wnt_x, g_wnt_y = self.cells.gradient(self.c_WNT)

        # Motor transport term:
        fx = -self.D_wnt * g_wnt_x
        fy = -self.D_wnt * g_wnt_y

        # divergence
        div_flux = self.cells.div(fx, fy, cbound=True)

        del_wnt = (-div_flux + rnai * self.r_wnt  * term_camp * self.NerveDensity -
                   self.d_wnt * self.c_WNT - self.d_wnt_deg_notum * term_notum * self.c_WNT* term_hh
                                           - self.d_wnt_deg_ptc * term_hh * self.c_WNT)

        return del_wnt  # change in Wnt

    def update_hh(self, rnai=1.0, kinesin=1.0):
        """
        Method describing change in Hedgehog levels in space and time.
        """

        # Gradient of concentration
        _, g_hh_x, g_hh_y = self.cells.gradient(self.c_HH)

        #         Motor transport term:
        m_hh = self.cells.meanval(self.c_HH)

        # Motor transport term:
        conv_term_x = m_hh * self.ux * self.u_hh * kinesin
        conv_term_y = m_hh * self.uy * self.u_hh * kinesin

        fx = -g_hh_x * self.D_hh + conv_term_x
        fy = -g_hh_y * self.D_hh + conv_term_y

        # fx = -g_hh_x * self.D_hh
        # fy = -g_hh_y * self.D_hh

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

        self.tot_conc_time = np.asarray(
            [(ci * self.cells.cell_vol).sum() for ci in self.c_Test_time]
        )

    # Post-processing functions--------------------------------
    def get_tops(self, cinds):
        """
        Collects the top 33% of the sample and averages it.
        :param cinds:
        :return:
        """

        top33 = int(len(cinds) / 3)

        sort_H = np.sort(self.Head[cinds])[::-1]
        sort_T = np.sort(self.Tail[cinds])[::-1]
        sort_B = np.sort(self.Blast[cinds])[::-1]

        pH = sort_H[:top33].mean()
        pT = sort_T[:top33].mean()
        pB = sort_B[:top33].mean()

        return pH, pT, pB

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
        mol_clims['Head'] = (0, 1.0)
        mol_clims['Tail'] = (0, 1.0)

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
        mol_cmaps['Head'] = cm.RdBu_r
        mol_cmaps['Tail'] = cm.magma

        self.default_cmaps = mol_cmaps

        self.groupc = ['LightBlue', 'DarkCyan', 'CadetBlue', 'SteelBlue', 'DeepSkyBlue', 'MediumBlue', 'DarkBlue',
                       'IndianRed', 'FireBrick', 'DarkRed', 'Brown', 'Green', 'Purple', 'Indigo', 'DarkSeaGreen']

    def triplot(self, ti, plot_type='init',
                autoscale=True, fname = 'Triplot_', dirsave=None, reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 8), axisoff=False, linew = 3.0,
                ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps


        #Filesaving:
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
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]
            carray3 = self.molecules_sim_time['Notum'][ti]


        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=fsize)

        if axisoff is True:

            ax1.yaxis.set_ticklabels([])
            ax2.yaxis.set_ticklabels([])
            ax3.yaxis.set_ticklabels([])

        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax3.xaxis.set_ticklabels([])

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

        ax2.add_collection(col2)
        ax2.set_title('β-Cat')
        ax2.axis('tight')

        ax3.add_collection(col3)
        ax3.set_title('Notum')
        ax3.axis('tight')

        if axisoff:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=ax1.transAxes)

        fig.subplots_adjust(wspace=0.0)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def hexplot(self, ti, plot_type='init', autoscale=True, fname = 'Hexplot_', dirsave=None, reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 14), axisoff=False, linew = 3.0,
                ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps


        #Filesaving:
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
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]
            carray3 = self.molecules_time['Notum'][ti]
            carray4 = self.molecules_time['Hh'][ti]
            carray5 = self.molecules_time['Wnt'][ti]
            carray6 = self.molecules_time['NRF'][ti]


        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]
            carray3 = self.molecules_time2['Notum'][ti]
            carray4 = self.molecules_time2['Hh'][ti]
            carray5 = self.molecules_time2['Wnt'][ti]
            carray6 = self.molecules_time2['NRF'][ti]


        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]
            carray3 = self.molecules_sim_time['Notum'][ti]
            carray4 = self.molecules_sim_time['Hh'][ti]
            carray5 = self.molecules_sim_time['Wnt'][ti]
            carray6 = self.molecules_sim_time['NRF'][ti]



        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(2, 3, figsize=fsize)

        axarr[0,0].xaxis.set_ticklabels([])
        axarr[0, 0].yaxis.set_ticklabels([])

        axarr[0, 1].xaxis.set_ticklabels([])
        axarr[0, 1].yaxis.set_ticklabels([])

        axarr[0, 2].xaxis.set_ticklabels([])
        axarr[0, 2].yaxis.set_ticklabels([])

        axarr[1, 0].xaxis.set_ticklabels([])
        axarr[1, 0].yaxis.set_ticklabels([])

        axarr[1, 1].xaxis.set_ticklabels([])
        axarr[1, 1].yaxis.set_ticklabels([])

        axarr[1, 2].xaxis.set_ticklabels([])
        axarr[1, 2].yaxis.set_ticklabels([])

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

        col4 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Hh'], linewidth=0.0)
        if autoscale is False:
            col4.set_clim(clims['Hh'][0], clims['Hh'][1])
        col4.set_array(carray4)


        col5 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Wnt'], linewidth=0.0)
        if autoscale is False:
            col5.set_clim(clims['Wnt'][0], clims['Wnt'][1])
        col5.set_array(carray5)


        col6 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['NRF'], linewidth=0.0)
        if autoscale is False:
            col6.set_clim(clims['NRF'][0], clims['NRF'][1])
        col6.set_array(carray6)


        axarr[0,0].add_collection(col1)
        axarr[0, 0].set_title('Erk')
        axarr[0, 0].axis('tight')
        axarr[0, 0].axis('off')

        axarr[0, 1].add_collection(col2)
        axarr[0, 1].set_title('β-Cat')
        axarr[0, 1].axis('tight')
        axarr[0, 1].axis('off')

        axarr[0, 2].add_collection(col3)
        axarr[0, 2].set_title('Notum')
        axarr[0, 2].axis('tight')
        axarr[0, 2].axis('off')

        axarr[1,0].add_collection(col4)
        axarr[1, 0].set_title('Hh')
        axarr[1, 0].axis('tight')
        axarr[1, 0].axis('off')

        axarr[1, 1].add_collection(col5)
        axarr[1, 1].set_title('Wnt')
        axarr[1, 1].axis('tight')
        axarr[1, 1].axis('off')

        axarr[1, 2].add_collection(col6)
        axarr[1, 2].set_title('NRF')
        axarr[1, 2].axis('tight')
        axarr[1, 2].axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0,0].transAxes)

        fig.subplots_adjust(wspace=0.0)

        plt.savefig(fname, format='png', dpi=reso,  transparent = True)
        plt.close()

    def biplot(self, ti, plot_type='init', autoscale=True, fname = 'Biplot_', dirsave=None, reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(10, 6), axisoff=False,
               ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if clims is None:
            clims = self.default_clims

        if cmaps is None:
            cmaps = self.default_cmaps

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
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['β-Cat'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]


        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]


        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        os.makedirs(dirstr, exist_ok=True)

        rcParams.update({'font.size': fontsize})

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=fsize)

        if axisoff is True:
            ax1.yaxis.set_ticklabels([])
            ax2.yaxis.set_ticklabels([])

        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])

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

        if axisoff:
            ax1.axis('off')

        ax2.add_collection(col2)
        ax2.axis('tight')
        ax2.set_title('β-Cat')

        if axisoff:
            ax2.axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=ax1.transAxes)

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


    def plot_frags(self, show_plot = False, save_plot = True,
                   reso = 150, group_colors = None, dir_save = None):

        if group_colors is None:
            group_colors = self.groupc

        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)

        for clustn, clusti in self.fragments.items():
            col1 = PolyCollection(self.verts_r[clusti] * 1e3, color=group_colors[clustn], edgecolor=None)
            ax.add_collection(col1)
            xi, yi = self.frag_xyr[clustn]
            plt.text(xi * 1e3, yi * 1e3, 'frag_' + str(clustn), color='black',
                     fontsize=18, fontweight='bold',
                     horizontalalignment='center', verticalalignment='center')

        plt.axis('equal')

        if show_plot:

            plt.show()

        if save_plot:

            fstr = 'IdentifiedCutFragments.png'

            if dir_save is None:
                dirstr = self.p.sim_export_dirname
                fname = os.path.join(self.p.sim_export_dirname, fstr)

            else:
                dirstr = dir_save
                fname = os.path.join(dirstr, fstr)

            os.makedirs(dirstr, exist_ok=True)

            plt.savefig(fname, format='png', dpi=reso, transparent=True)
            plt.close()

    def animate_triplot(self, ani_type='init', autoscale=True, dirsave=None, reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 8), axisoff=False, linew = 3.0,
                        ref_data=None, extra_text = None):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.triplot(ii, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso, linew = 3.0,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff,
                             ref_data=ref_data, extra_text = extra_text)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.triplot(ii, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso, linew = 3.0,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff,
                             ref_data=ref_data, extra_text = extra_text)

    def animate_biplot(self, ani_type='init', autoscale=True, dirsave=None, reso=150,
               clims=None, cmaps=None, fontsize=18.0, fsize=(10, 6), axisoff=False,
                       ref_data=None, extra_text = None):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.biplot(ii, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff,
                            ref_data=ref_data, extra_text = extra_text)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.biplot(ii, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff,
                            ref_data=ref_data, extra_text = extra_text)

    def animate_plot(self, ctag, ani_type='init', autoscale=True, dirsave='PlotAni', reso=150,
             clims=None, cmaps=None, fontsize=18.0, fsize=(4, 6), axisoff=False):

        if ani_type == 'init' or ani_type == 'reinit':

            for ii, ti in enumerate(self.tsample_init):
                self.plot(ii, ctag, plot_type='init', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

        elif ani_type == 'sim':

            for ii, ti in enumerate(self.tsample_sim):
                self.plot(ii, ctag, plot_type='sim', autoscale=autoscale, dirsave=dirsave, reso=reso,
                             clims = clims, cmaps=cmaps, fontsize=fontsize, fsize=fsize, axisoff = axisoff)

    def markovplot(self, ti, plot_type='init', autoscale=True, fname = 'Markov', dirsave=None, reso=150,
                clims=None, cmaps=None, fontsize=18.0, fsize=(6, 8), axisoff=False, linew = 3.0,
                ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        if cmaps is None:
            cmaps = self.default_cmaps


        #Filesaving:
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
            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['Head'][ti]
            carray3 = self.molecules_time['Tail'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit

            self.assign_easy_x(self.cells_i)
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['Head'][ti]
            carray3 = self.molecules_time2['Tail'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim

            self.assign_easy_x(self.cells_s)
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['Head'][ti]
            carray3 = self.molecules_sim_time['Tail'][ti]

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=fsize)

        if axisoff is True:

            ax1.yaxis.set_ticklabels([])
            ax2.yaxis.set_ticklabels([])
            ax3.yaxis.set_ticklabels([])

        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        ax3.xaxis.set_ticklabels([])

        col1 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Erk'], linewidth=0.0)
        if autoscale is False:
            col1.set_clim(0.0, 1.0)
        col1.set_array(carray1)


        col2 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Head'], linewidth=0.0)
        if autoscale is False:
            col2.set_clim(0.0, 1.0)
        col2.set_array(carray2)


        col3 = PolyCollection(self.verts_r * 1e3, edgecolor=None, cmap=cmaps['Tail'], linewidth=0.0)
        if autoscale is False:
            col3.set_clim(0.0, 1.0)
        col3.set_array(carray3)


        ax1.add_collection(col1)

        ax1.set_title('ERK')
        ax1.axis('tight')

        ax2.add_collection(col2)
        ax2.set_title('pHead')
        ax2.axis('tight')

        ax3.add_collection(col3)
        ax3.set_title('pTail')
        ax3.axis('tight')

        if axisoff is True:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string, x=0.5, y=0.1)

        fig.subplots_adjust(wspace=0.0)

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=ax1.transAxes)

        if plot_type == 'sim':

            # Add a data table to the plot describing outcome heteromorph probabilities:
            hmorph_data, col_names, row_names = self.heteromorph_table(transpose=False)

            # plt.tight_layout(rect=[0.0, 0.0, 0.5, 0.95])

            tabo = plt.table(cellText=hmorph_data, loc='right',
                      cellLoc='left', rowLoc='left', colLoc='left', colLabels=col_names,
                      rowLabels=row_names, edges='open', bbox=[1.5, 0.05, 2.0, 1.0])
            tabo.auto_set_font_size(False)
            tabo.set_fontsize(fontsize - 4)

        plt.savefig(fname, format='png', dpi=reso, bbox_inches='tight')
        plt.close()




