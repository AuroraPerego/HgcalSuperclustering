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

from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties, CPtoCandidate_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *

from numba import prange
import multiprocessing as mp

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 = "/eos/home-a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo/"
fileV4 = "/eos/home-a/aperego/SampleProduction/TICLv4/ParticleGunPionPU/histo/"

OutputDir = "/eos/user/a/aperego/www/PerformanceTICLv5/resolution/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
], limitFileCount=None)


dumperInputV4 = DumperInputManager([
    fileV4
], limitFileCount=None)

def doEverything(dumperInput, isV5 = True):
    time_chg = []
    time_neut = []
    time_chg_sigma = []
    time_neut_sigma = []
    for i in prange(len(dumperInput.inputReaders)):
        dumper = dumperInput.inputReaders[i].ticlDumperReader
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        sims = dumper.simTrackstersCP
        print("file ", i, "Nevents ", len(cands))
        for ev in prange(len(cands)):
            candEv = cands[ev]
            simCandEv = simCands[ev]
            assEv = ass[ev]
            simEv = sims[ev]

            for cand_idx in prange(len(candEv.candidate_raw_energy)):
                time = candEv.candidate_time[cand_idx]
                timeErr = candEv.candidate_timeErr[cand_idx]
                recoTrack = candEv.track_in_candidate[cand_idx]
                ts_idx = candEv.tracksters_in_candidate[cand_idx]
                rawEnergy = candEv.candidate_raw_energy[cand_idx]

                if len(ts_idx) == 0:
                    continue

                if isV5:
                    ts_idx = ts_idx[0]
                    recoToSim = assEv.ticlCandidate_recoToSim_CP[ts_idx]
                    sharedE = assEv.ticlCandidate_recoToSim_CP_sharedE[ts_idx]
                    score = assEv.ticlCandidate_recoToSim_CP_score[ts_idx]
                else:
                    recoToSim = assEv.Mergetracksters_recoToSim_CP[cand_idx]
                    sharedE = assEv.Mergetracksters_recoToSim_CP_sharedE[cand_idx]
                    score = assEv.Mergetracksters_recoToSim_CP_score[cand_idx]
                if not len(sharedE): continue
                argminScore = ak.argmin(score)
#                 print(rawEnergy, cand_idx, recoToSim, recoToSim[argminScore])
                simCand_idx = recoToSim[argminScore]
                if simCand_idx > 1:
                    continue
                simTime = simCandEv.simTICLCandidate_time[simCand_idx]
                simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
                if sharedE[argminScore]/simRawEnergy < 0.1: continue

                if timeErr > 0:
                    if recoTrack != -1:
                        time_chg.append(time - simTime)
                        time_chg_sigma.append((time - simTime)/timeErr)
                    else:
                        time_neut.append(time - simTime)
                        time_neut_sigma.append((time - simTime)/timeErr)

    return time_chg, time_chg_sigma, time_neut, time_neut_sigma

# time_chg, time_chg_sigma, time_neut, time_neut_sigma = doEverything(dumperInputV5)

# time_chg_v4, time_chg_sigma_v4, time_neut_v4, time_neut_sigma_v4 = doEverything(dumperInputV4, False)

if __name__ == '__main__':  # Ensure the multiprocessing code runs only when executed directly
    # Create a multiprocessing pool with 2 processes
    with mp.Pool(processes=8) as pool:
        # Launch both tasks in parallel
        future_v5 = pool.apply_async(doEverything, args=(dumperInputV5, True))  # Flag defaults to True
        future_v4 = pool.apply_async(doEverything, args=(dumperInputV4, False))  # Pass flag=False

        # Wait for both results to complete and retrieve them
        time_chg, time_chg_sigma, time_neut, time_neut_sigma = future_v5.get()  # get() blocks until done
        time_chg_v4, time_chg_sigma_v4, time_neut_v4, time_neut_sigma_v4 = future_v4.get()

        # #########
        # # PLOTS #
        # #########

        plt.figure()
        hep.cms.text("Simulation Preliminary", loc=0)
        plt.text(2500, -0.4, r"$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
        plt.hist(time_chg, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'red', label="TICL V5")
        plt.hist(time_chg_v4, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'blue', label="current")
        plt.legend()
        plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$ [ns]')
        plt.ylabel("Entries")
        plt.savefig(OutputDir + "charged_time_res.png")

        plt.figure()
        hep.cms.text("Simulation Preliminary", loc=0)
        plt.text(14, -0.4, r"$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
        plt.hist(time_chg, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'red', label="TICL V5", density=True)
        plt.hist(time_chg_v4, range=(-0.4, 0.4), bins = 30, histtype = "step", lw = 2, color = 'blue', label="current", density=True)
        plt.legend()
        plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$ [ns]')
        plt.ylabel("Entries")
        plt.savefig(OutputDir + "charged_time_res_densityTrue.png")

        plt.figure()
        hep.cms.text("Simulation Preliminary", loc=0)
        plt.text(1100, -5, r"$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
        plt.hist(time_chg_sigma, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICL V5")
        plt.hist(time_chg_sigma_v4, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="current")
        plt.legend()
        plt.xlabel(r'$\frac{t_{reco} - t_{sim}^{vtx}}{\sigma_t}$')
        plt.savefig(OutputDir + "charged_time_res_sigma.png")


        plt.figure()
        hep.cms.text("Simulation Preliminary", loc=0)
        plt.text(1800, -0.5, r"$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
        plt.hist(time_neut, range=(-0.5, 0.5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICL V5")
        plt.hist(time_neut_v4, range=(-0.5, 0.5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="current")
        plt.legend()
        plt.xlabel(r'$t_{reco} - t_{sim}^{vtx}$')
        plt.savefig(OutputDir + "neut_time_res.png")

        plt.figure()
        hep.cms.text("Simulation Preliminary", loc=0)
        plt.text(400, -5, r"$\pi^{\pm}$ 200PU\n$1<p_T<100$ GeV\n$1.7<|\eta|<2.7$")
        plt.hist(time_neut_sigma, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'red', label="TICL V5")
        plt.hist(time_neut_sigma_v4, range=(-5, 5), bins = 20, histtype = "step", lw = 2, color = 'blue', label="current")
        plt.legend()
        plt.xlabel(r'$\frac{t_{reco} - t_{sim}^{vtx}}{\sigma_t}$')
        plt.savefig(OutputDir + "neut_time_res_sigma.png")

        np.savetxt("dataResidui/time_chg.txt", time_chg)
        np.savetxt("dataResidui/time_chg_sigma.txt", time_chg_sigma)
        np.savetxt("dataResidui/time_neut.txt", time_neut)
        np.savetxt("dataResidui/time_neut_sigma.txt", time_neut_sigma)
        np.savetxt("dataResidui/time_chg_v4.txt", time_chg_v4)
        np.savetxt("dataResidui/time_chg_sigma_v4.txt", time_chg_sigma_v4)
        np.savetxt("dataResidui/time_neut_v4.txt", time_neut_v4)
        np.savetxt("dataResidui/time_neut_sigma_v4.txt", time_neut_sigma_v4)

