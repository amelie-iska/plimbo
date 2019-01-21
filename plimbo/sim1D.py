#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
1D Simulator object for Planarian Interface for PLIMBO module.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import OrderedDict
import os
import os.path
from betse.science.math import toolbox as tb
from plimbo.simabc import PlanariaGRNABC
from sklearn.cluster import DBSCAN


class PlanariaGRN1D(PlanariaGRNABC):
    """
    Object describing 1D version of core GRN model.
    A BETSE config file is used to define paths for saving image and data exports.

    """

    def __init__(self, *args, **kwargs):

        self.model_init(*args, **kwargs)

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

        # Short x-axis:
        self.Xs = np.dot(self.Mx, self.X)

        # dummy cells object:
        self.cells = None

    def run_markov(self, ti):
        """
        Updates the Markov model in time

        :return:

        """

        gpulse = 1.0

        if self.runtype == 'sim': # limit growth to wounds for a timed process:
            gpulse = 1.0 - tb.step(ti, self.hdac_to, self.hdac_ts)

        elif self.runtype == 'reinit':
            gpulse = 0.0 # inhibit growth of hdac

        # update transition constants based on new value of ERK and beta-Cat:
        self.alpha_BH = 1/(1 + np.exp(-(self.c_ERK - self.C1)/self.K1)) # init transition constant blastema to head
        self.alpha_BT = 1/(1 + np.exp(-(self.c_BC - self.C2)/self.K2)) # init transition constant blastema to tail

        # update probabilities using an Implicit Euler Scheme:
        dt = self.dt * gpulse

        denom = (self.beta_B * dt + 1) * (self.beta_B * dt + self.alpha_BT * dt + self.alpha_BH * dt + 1)

        self.Tail = (
                    self.alpha_BT * self.beta_B * dt ** 2 + self.Tail * self.beta_B * dt - self.Head * self.alpha_BT * dt +
                    self.alpha_BT * dt + self.Tail * self.alpha_BH * dt + self.Tail) / denom

        self.Head = (
                    self.alpha_BH * self.beta_B * dt ** 2 + self.Head * self.beta_B * dt + self.Head * self.alpha_BT * dt -
                    self.Tail * self.alpha_BH * dt + self.alpha_BH * dt + self.Head) / denom

        self.Blast = 1.0 - self.Head - self.Tail

    def load_transport_field(self):

        # Transport fields
        self.K_u = 0.5
        self.n_u = 0.5

        self.u = 1 / (1 + (self.Xs / (self.xmid / self.K_u)) ** self.n_u)

        NerveDensity = 1 / (1 + (self.X / (self.xmid / self.K_u)) ** self.n_u)

        NDmin = NerveDensity.min()
        NDmax = NerveDensity.max()

        # Normalize the nerve map so that it ranges from user-specified  min to max values:
        self.NerveDensity = (NerveDensity - NDmin) * ((self.n_max - self.n_min) / (NDmax - NDmin)) + self.n_min

        # print("The mean nerve density is: ", self.NerveDensity.mean())

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

    def cut_cells(self):

        if self.model_has_been_cut is False:

            Xcut = np.delete(self.X, self.cut_inds)
            Ycut = np.zeros(Xcut.shape)

            self.XY = np.column_stack((Xcut, Ycut))

            self.new_cdl = len(self.XY)

            self.all_inds = np.linspace(0, self.new_cdl - 1, self.new_cdl, dtype=np.int)

            hurt_mask = np.zeros(self.cdl)
            hurt_mask[self.target_inds_wound] = 1.0
            hurt_mask = np.delete(hurt_mask, self.cut_inds)
            self.hurt_inds = (hurt_mask == 1.0).nonzero()[0]

                    # identify clusters of indices representing each fragment:
            self.fragments, self.frag_xy = self.cluster_points(self.all_inds, dmax = 2.0)

            # organize the data:
            wounds = OrderedDict()  # dictionary holding indices of clusters
            wounds_xy = OrderedDict()  # dictionary of cluster x,y point centers

            for ii, wndi in enumerate(self.hurt_inds):
                wounds[ii] = wndi
                wounds_xy[ii] = self.XY[wndi]

            # Organize wounds into fragments:
            self.frags_and_wounds = OrderedDict()

            for fragi in self.fragments.keys():
                self.frags_and_wounds[fragi] = []

            for fragn, fragi in self.fragments.items():

                for woundn, woundi in wounds.items():

                    intersecto = np.intersect1d(woundi, fragi)

                    if len(intersecto):
                        self.frags_and_wounds[fragn].append(intersecto)

            self.model_has_been_cut = True

    def cluster_points(self, cinds, dmax=2.0, min_samples = 2):
        """
        Identifies clusters of points (e.g. fragment or wounded zones within a fragment)

        cinds: indices to self.cells.cell_centers array, or a subset (typically self.cells.cell_i)
        dmax: multiplier of self.p.cell_radius (maximum nearest-neighbour distance)
\
        """

        maxd = dmax*self.p.cell_radius*self.x_scale  # maximum search distance to nearest-neighbours

        cell_centres = self.XY[cinds]  # get relevant cloud of x,y points to cluster

        cell_i = self.all_inds[cinds]  # get relevant indices for the case we're working with a subset

        clust = DBSCAN(eps=maxd, min_samples=min_samples).fit(cell_centres)  # Use scikitlearn to flag clusters

        # organize the data:
        clusters = OrderedDict()  # dictionary holding indices of clusters
        cluster_xy = OrderedDict()  # dictionary of cluster x,y point centers

        for li in clust.labels_:
            clusters[li] = []

        for ci, li in zip(cell_i, clust.labels_):
            clusters[li].append(ci)

        for clustn, clusti in clusters.items():
            pts = self.XY[clusti]

            cluster_xy[clustn] = (pts[0].mean(), pts[1].mean())

        return clusters, cluster_xy

    def scale_cells(self, x_scale):

        pass

    # GRN Updating functions---------------------------------------

    def update_bc(self, rnai_bc=1.0, rnai_apc=1.0, rnai_camp = 1.0, rnai_erk =1.0):
        """
        Method describing change in beta-cat levels in space and time.
        """

        iWNT = (self.c_WNT / self.K_apc_wnt) ** self.n_apc_wnt
        term_wnt_i = 1 / (1 + iWNT) # Wnt inhibits activation of the APC-based degradation complex

        icAMP = ((rnai_camp*self.c_cAMP)/self.K_bc_camp) ** self.n_bc_camp
        term_camp_i = 1 / (1 + icAMP) # cAMP inhibits activity of the APC by inhibiting activation of APC
        term_camp_g = icAMP / (1 + icAMP) # cAMP inhibits activity of the APC by promoting deactivation of APC

        # calculate steady-state of APC activation:
        cAPC = (term_wnt_i*term_camp_i)/term_camp_g

        # calculate APC-mediated degradation of beta-Cat:
        apci = (cAPC/self.K_bc_apc)**self.n_bc_apc
        term_apc = apci/(1 + apci)

        # gradient and midpoint mean concentration:
        g_bc, m_bc = self.get_gradient(self.c_BC, self.runtype)

        flux = -g_bc * self.D_bc

        # divergence of the flux
        div_flux = self.get_div(flux, self.runtype)

        # change of bc:
        del_bc = (-div_flux + rnai_bc*self.r_bc  - self.d_bc*self.c_BC
                  - rnai_apc*self.d_bc_deg*term_apc*self.c_BC)

        # update ERK, assuming rapid change in signalling compared to gene expression:
        iBC = (self.c_BC / self.K_erk_bc) ** self.n_erk_bc
        term_bc = 1 / (1 + iBC)

        self.c_ERK = rnai_erk*term_bc

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
        del_nrf = (-div_flux + self.r_nrf * term_bc*self.NerveDensity - self.d_nrf * self.c_NRF)

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

    def update_wnt(self, rnai_wnt=1.0, rnai_ptc = 1.0):
        """
        Method describing combined change in Wnt1 and Wnt 11 levels in space and time.
        """

        # Growth and decay
        iNotum = (self.c_Notum / self.K_wnt_notum) ** self.n_wnt_notum
        term_notum = iNotum / (1 + iNotum) # Notum promotes decay of Wnt1

        iHH = (self.c_HH / self.K_wnt_hh) ** self.n_wnt_hh
        term_hh = 1 / (1 + iHH) # HH inhibits decay of Wnt11 via Ptc

        dptc = self.d_wnt_deg_ptc*term_hh*rnai_ptc # decay of Wnt11 via Ptc
        dnot = self.d_wnt_deg_notum*term_notum # decay of Wnt1 via Notum

        # effective decay rate of wnt1 + wnt11 combo (where Notum acts on Wnt1 and Ptc acts on Wnt11):
        # ndense = self.NerveDensity
        # effective_d = ((dnot + self.d_wnt)*(dptc + self.d_wnt))/(self.d_wnt + dptc + dnot + self.d_wnt)

        # effective decay rate of wnt1 + wnt11 combo (where Notum acts on all Wnts and Ptc acts on Wnt11):
        # gm = self.NerveDensity # growth modulator for Wnt1
        # gm = 1.0

        # effective_d = ((dnot + self.d_wnt)*(dnot + dptc + self.d_wnt))/(self.d_wnt*(1 + gm) + dptc*gm + dnot*(1 + gm))

        # Gradient and mean of concentration
        g_wnt, m_wnt = self.get_gradient(self.c_WNT, self.runtype)

        # Transport flux:
        flux = -self.D_wnt * g_wnt

        # divergence
        div_flux = self.get_div(flux, self.runtype)

        # effective change in the combined concentration of Wnt1 and Wnt11:
        # del_wnt = -div_flux + rnai_wnt*self.r_wnt - effective_d*self.c_WNT
        del_wnt = -div_flux + rnai_wnt*self.r_wnt - (dnot + dptc + self.d_wnt)*self.c_WNT


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
        del_hh = (-div_flux + rnai*self.r_hh*self.NerveDensity - self.d_hh * self.c_HH)

        return del_hh  # change in Hedgehog

    def get_tops(self, cinds):
        """
        Collects the sample at the wound indices and averages it.
        :param cinds:
        :return:
        """

        Headx = np.delete(self.Head, self.cut_inds)
        Tailx = np.delete(self.Tail, self.cut_inds)
        Blastx = np.delete(self.Blast, self.cut_inds)

        pH = Headx[cinds].mean()
        pT = Tailx[cinds].mean()
        pB = Blastx[cinds].mean()

        return pH, pT, pB

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
        mol_clims['NotumRNA'] = [0, 1.0]
        mol_clims['APC'] = [0, 1.0]
        mol_clims['cAMP'] = [0, 1.0]
        mol_clims['Head'] = [0, 1.0]
        mol_clims['Tail'] = [0, 1.0]

        self.default_clims = mol_clims

        mol_cmaps = OrderedDict()

        mol_cmaps['β-Cat'] = 'Blue'
        mol_cmaps['Erk'] = 'Red'
        mol_cmaps['Wnt'] = 'DodgerBlue'
        mol_cmaps['Hh'] = 'DarkCyan'
        mol_cmaps['NRF'] = 'Blue'
        mol_cmaps['Notum'] = 'Green'
        mol_cmaps['NotumRNA'] = 'Green'
        mol_cmaps['APC'] = 'OrangeRed'
        mol_cmaps['cAMP'] = 'DeepSkyBlue'
        mol_cmaps['Tail'] = 'Blue'
        mol_cmaps['Head'] = 'Red'

        self.default_cmaps = mol_cmaps


    def triplot(self, ti, plot_type = 'init',
                fname = 'Triplot_', dirsave = None, reso = 150, linew = 3.0, axisoff = False,
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

            # plot the relative rate of Notum transcription instead of Notum
            # iNRF = (self.molecules_time['NRF'][ti]/ self.K_notum_nrf) ** self.n_notum_nrf
            # carray3 = iNRF / (1 + iNRF)
            carray3 = self.molecules_time['Notum'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['β-Cat'][ti]

            # plot the relative rate of Notum transcription instead of Notum
            # iNRF = (self.molecules_time2['NRF'][ti]/ self.K_notum_nrf) ** self.n_notum_nrf
            # carray3 = iNRF / (1 + iNRF)
            carray3 = self.molecules_time2['Notum'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['β-Cat'][ti]

            # plot the relative rate of Notum transcription instead of Notum
            # iNRF = (self.molecules_sim_time['NRF'][ti]/ self.K_notum_nrf) ** self.n_notum_nrf
            # carray3 = iNRF / (1 + iNRF)
            carray3 = self.molecules_sim_time['Notum'][ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)
            _, cs3 = self.get_plot_segs(carray3)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            # main plot data:
            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)
            axarr[2].plot(self.X*1e3, carray3, color=cmaps['NotumRNA'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                cNRFr = ref_data['NRF'][ti]
                iNRF = (cNRFr / self.K_notum_nrf) ** self.n_notum_nrf
                carray3r = iNRF / (1 + iNRF)

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)
                axarr[2].plot(self.X * 1e3, carray3r, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

        elif plot_type == 'sim':

            # main plot data
            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

            for xi, ci in zip(xs, cs3):
                axarr[2].plot(xi, ci, color=cmaps['NotumRNA'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                cNRFr = ref_data['NRF'][ti]
                iNRF = (cNRFr / self.K_notum_nrf) ** self.n_notum_nrf
                carray3r = iNRF / (1 + iNRF)

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)
                _, cs3r = self.get_plot_segs(carray3r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

                for xi, ci in zip(xsr, cs3r):
                    axarr[2].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)


        axarr[0].set_title("ERK")
        axarr[0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[0].set_ylim(clims['Erk'][0], clims['Erk'][1])

        axarr[1].set_title("β-cat")
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

        plt.savefig(fname, format='png', dpi=reso, transparent = True)
        plt.close()

    def biplot(self, ti, plot_type = 'init', fname = 'Biplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
               ref_data=None, extra_text = None, txt_x = 0.05, txt_y = 0.92, axisoff = False):


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

            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['β-Cat'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

        elif plot_type == 'sim':

            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['β-Cat'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['β-Cat'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed', zorder =10)

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

    def plot(self, ti, ctag, plot_type='init', dirsave = 'Plot', reso = 150, linew = 3.0, axisoff = False,
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

    def animate_triplot(self, ani_type = 'init', dirsave = None, reso = 150, linew = 3.0, axisoff = False,
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


    def animate_biplot(self, ani_type='init', dirsave = None, reso = 150, linew = 3.0, axisoff = False,
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

    def animate_plot(self, ctag, ani_type='init', dirsave = 'PlotAni', reso = 150, linew = 3.0, axisoff = False,
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
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92, axisoff = False):

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
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92, axisoff = False):

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
            carray1 = self.molecules_time['Erk'][ti]
            carray2 = self.molecules_time['Head'][ti]
            carray3 = self.molecules_time['Tail'][ti]

        elif plot_type == 'reinit':

            tsample = self.tsample_reinit
            carray1 = self.molecules_time2['Erk'][ti]
            carray2 = self.molecules_time2['Head'][ti]
            carray3 = self.molecules_time2['Tail'][ti]

        elif plot_type == 'sim':
            tsample = self.tsample_sim
            carray1 = self.molecules_sim_time['Erk'][ti]
            carray2 = self.molecules_sim_time['Head'][ti]
            carray3 = self.molecules_sim_time['Tail'][ti]

            xs, cs1 = self.get_plot_segs(carray1)
            _, cs2 = self.get_plot_segs(carray2)
            _, cs3 = self.get_plot_segs(carray3)

        else:
            print("Valid plot types are 'init', 'reinit', and 'sim'.")

        rcParams.update({'font.size': fontsize})

        fig, axarr = plt.subplots(3, sharex=True, figsize=fsize)

        if plot_type == 'init' or plot_type == 'reinit':

            # main plot data:
            axarr[0].plot(self.X*1e3, carray1, color=cmaps['Erk'], linewidth=linew)
            axarr[1].plot(self.X*1e3, carray2, color=cmaps['Head'], linewidth=linew)
            axarr[2].plot(self.X*1e3, carray3, color=cmaps['Tail'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew*0.5 # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['Head'][ti]
                carray3r = ref_data['Tail'][ti]

                axarr[0].plot(self.X * 1e3, carray1r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[1].plot(self.X * 1e3, carray2r, color='Black', linewidth=linewr, linestyle='dashed')
                axarr[2].plot(self.X * 1e3, carray3r, color='Black', linewidth=linewr, linestyle='dashed')


        elif plot_type == 'sim':

            # main plot data
            for xi, ci in zip(xs, cs1):
                axarr[0].plot(xi, ci, color=cmaps['Erk'], linewidth=linew)

            for xi, ci in zip(xs, cs2):
                axarr[1].plot(xi, ci, color=cmaps['Head'], linewidth=linew)

            for xi, ci in zip(xs, cs3):
                axarr[2].plot(xi, ci, color=cmaps['Tail'], linewidth=linew)

            if ref_data is not None:  # if a reference line is supplied, prepare it for the plot

                linewr = linew * 0.5  # make the reference line a bit thinner

                carray1r = ref_data['Erk'][ti]
                carray2r = ref_data['Head'][ti]
                carray3r = ref_data['Tail'][ti]

                xsr, cs1r = self.get_plot_segs(carray1r)
                _, cs2r = self.get_plot_segs(carray2r)
                _, cs3r = self.get_plot_segs(carray3r)

                for xi, ci in zip(xsr, cs1r):
                    axarr[0].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs2r):
                    axarr[1].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')

                for xi, ci in zip(xsr, cs3r):
                    axarr[2].plot(xi, ci, color='Black', linewidth=linewr, linestyle='dashed')


        axarr[0].set_title("Erk")
        axarr[0].set_ylabel('Concentration [nM]')
        if autoscale is False:
            axarr[0].set_ylim(0, 1)

        axarr[1].set_title("pHead")
        axarr[1].set_ylabel('Probability')
        if autoscale is False:
            axarr[1].set_ylim(0, 1)

        axarr[2].set_title("pTail")
        axarr[2].set_ylabel('Probability')
        if autoscale is False:
            axarr[2].set_ylim(0, 1)

        axarr[2].set_xlabel('Axis Distance [mm]')

        if extra_text is not None:
            fig.text(txt_x, txt_y, extra_text, transform=axarr[0].transAxes)

        fig.subplots_adjust(hspace=0.15)

        if plot_type == 'sim':

            # Add a data table to the plot describing outcome heteromorph probabilities:
            hmorph_data, col_names, row_names = self.heteromorph_table(transpose=True)

            plt.tight_layout(rect=[0.04, 0.25, 0.95, 0.95])

            plt.table(cellText=hmorph_data, loc='bottom',
                      cellLoc='left', rowLoc='left', colLoc='left', colLabels = col_names,
                      rowLabels=row_names, edges='open', bbox=[0, -1.7, 1, 1.2])


        fig.suptitle('Initialization', x=0.1, y=0.94)

        tt = tsample[ti]

        tdays = np.round(tt / (3600), 1)
        tit_string = str(tdays) + ' Hours'
        fig.suptitle(tit_string)

        plt.savefig(fname, format='png', dpi=reso, transparent = False)
        plt.close()

    def plot_frags(self, show_plot = False, save_plot = True, fsize=(12, 12),
                   reso = 150, group_colors = None, dir_save = None):

        pass













