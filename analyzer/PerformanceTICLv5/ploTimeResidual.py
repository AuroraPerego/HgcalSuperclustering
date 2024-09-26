import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

#plt.rcParams['text.usetex'] = True

plt.style.use(hep.style.CMS)
import hist

from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *


time_chg = np.loadtxt("dataResidui/time_chg.txt")
time_chg_sigma = np.loadtxt("dataResidui/time_chg_sigma.txt")
time_neut = np.loadtxt("dataResidui/time_neut.txt")
time_neut_sigma = np.loadtxt("dataResidui/time_neut_sigma.txt")
time_chg_v4 = np.loadtxt("dataResidui/time_chg_v4.txt")
time_chg_sigma_v4 = np.loadtxt("dataResidui/time_chg_sigma_v4.txt")
time_neut_v4 = np.loadtxt("dataResidui/time_neut_v4.txt")
time_neut_sigma_v4 = np.loadtxt("dataResidui/time_neut_sigma_v4.txt")

OutputDir = "/eos/user/a/aperego/www/PerformanceTICLv5/resolution_new/"

# #########
# # PLOTS #
# #########

plt.figure()
hep.cms.text("Simulation Preliminary", loc=0)
plt.text(-0.4, 2000, "$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
plt.hist(time_chg, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'red', label="TICLv5a")
plt.hist(time_chg_v4, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'blue', label="Current")
plt.legend()
plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$ [ns]')
plt.ylabel("Entries")
plt.savefig(OutputDir + "charged_time_res.png")

plt.figure()
hep.cms.text("Simulation Preliminary", loc=0)
plt.text(-0.4, 12, "$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
plt.hist(time_chg, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'red', label="TICLv5a", density=True)
plt.hist(time_chg_v4, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'blue', label="Current", density=True)
plt.legend()
plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$ [ns]')
plt.ylabel("Entries")
plt.savefig(OutputDir + "charged_time_res_densityTrue.png")

plt.figure()
hep.cms.text("Simulation Preliminary", loc=0)
plt.text(-5, 900, "$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
plt.hist(time_chg_sigma, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICLv5a")
plt.hist(time_chg_sigma_v4, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="Current")
plt.legend()
plt.xlabel(r'$\frac{t_{reco} - t_{sim}^{vtx}}{\sigma_t}$')
plt.savefig(OutputDir + "charged_time_res_sigma.png")


plt.figure()
hep.cms.text("Simulation Preliminary", loc=0)
plt.text(-0.5, 1500, "$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
plt.hist(time_neut, range=(-0.5, 0.5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICLv5a")
plt.hist(time_neut_v4, range=(-0.5, 0.5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="Current")
plt.legend()
plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$')
plt.savefig(OutputDir + "neut_time_res.png")

plt.figure()
hep.cms.text("Simulation Preliminary", loc=0)
plt.text(-5, 300, "$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
plt.hist(time_neut_sigma, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICLv5a")
plt.hist(time_neut_sigma_v4, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="Current")
plt.legend()
plt.xlabel(r'$\frac{t_{reco} - t_{sim}^{vtx}}{\sigma_t}$')
plt.savefig(OutputDir + "neut_time_res_sigma.png")

