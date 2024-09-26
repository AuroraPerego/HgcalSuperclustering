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

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 = "/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo/"
fileV5_noTime = "/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU_noTime/histo/"
# fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/TimeResolution/SinglePionTiming_2p2_100GeV/histo/"

OutputDir = "/eos/user/a/aperego/www/PerformanceTICLv5_strict/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
], limitFileCount=35)


dumperInputV5_noTime = DumperInputManager([
    fileV5_noTime
], limitFileCount=35)
'''
print(CPtoCandidate_properties)
resV5 = runComputations([CPtoCandidate_properties], dumperInputV5, max_workers=1)
print(resV5)


dumper = dumperInputV5.inputReaders[0].ticlDumperReader
cands = dumper.candidates

print(cands.candidate_raw_energy[0])
print(cands.candidate_raw_energy[1])
print(cands.candidate_raw_energy[99])

a = ak.Array([ak.Array([x[0] if len(x) > 0 else -1 for x in y]) for y in cands.tracksters_in_candidate])
print(a, a[0])
b = ak.Array([x[0] if len(x) > 0 else -1 for x in subarray] for subarray in cands.tracksters_in_candidate)
print(b ,b[0])
'''

def doEverything(dumperInput):
    alls = []
    with_track = []
    efficients = []
    denominator = []
    numerator = []

    pt_all = []
    pt_with_track = []
    pt_efficients = []
    pt_denominator = []
    pt_numerator = []

    fake_all = []
    fake_with_track = []
    fake_denominator = []
    fake_numerator = []
    fake_numerator_strict = []

    fake_pt_all = []
    fake_pt_with_track = []
    fake_pt_denominator = []
    fake_pt_numerator = []
    fake_pt_numerator_strict = []
    for i in prange(len(dumperInput.inputReaders)):
        dumper = dumperInput.inputReaders[i].ticlDumperReader
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        sims = dumper.simTrackstersCP
        tracks = dumper.tracks
        print("file ", i, "Nevents ", len(cands))
        for ev in prange(len(cands)):
            candEv = cands[ev]
            simCandEv = simCands[ev]
            assEv = ass[ev]
            simEv = sims[ev]
            tracksEv = tracks[ev]
            for simCand_idx in prange(len(simCandEv.simTICLCandidate_raw_energy)):
                simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
                simRegrEnergy = simCandEv.simTICLCandidate_regressed_energy[simCand_idx]
                simTrack = simCandEv.simTICLCandidate_track_in_candidate[simCand_idx]
                simPt = simEv.raw_pt[simCand_idx]
    
                alls.append(simRegrEnergy)
                pt_all.append(simPt)
                
                tk_idx_in_coll = -1
                try:
                    tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
                except:
                    continue
                if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                    continue
    
                with_track.append(simRegrEnergy)
                pt_with_track.append(simPt)
                
                print(assEv.ticlCandidate_simToReco_CP, assEv.ticlCandidate_simToReco_CP_sharedE, assEv.ticlCandidate_simToReco_CP_score)
                simToReco = assEv.ticlCandidate_simToReco_CP[simCand_idx]
                sharedE = assEv.ticlCandidate_simToReco_CP_sharedE[simCand_idx]
                score = assEv.ticlCandidate_simToReco_CP_score[simCand_idx]
                if not len(sharedE): continue
                if sharedE[ak.argmin(score)]/simRegrEnergy < 0.5: continue
                tid = simToReco[ak.argmin(score)]
    
                efficients.append(simRegrEnergy)
                pt_efficients.append(simPt)
                
                # obtain cand idx
                cand_idx = -1
                for i, k in enumerate(candEv.tracksters_in_candidate):
                    if not len(k): continue
                    if k[0] == tid:
                        cand_idx = i
                        break           
                if cand_idx == -1: continue
    
                denominator.append(simRegrEnergy)
                pt_denominator.append(simPt)
    
                recoTrack = candEv.track_in_candidate[cand_idx]
                if recoTrack == simTrack:
                    # num += 1
                    numerator.append(simRegrEnergy)
                    pt_numerator.append(simPt)

            for cand_idx in prange(len(candEv.candidate_raw_energy)):
                rawEnergy = candEv.candidate_raw_energy[cand_idx]
                regrEnergy = candEv.candidate_energy[cand_idx]
                recoTrack = candEv.track_in_candidate[cand_idx]
                recoPt = (candEv.candidate_px[cand_idx]**2 + candEv.candidate_py[cand_idx]**2)**0.5
                ts_idx = candEv.tracksters_in_candidate[cand_idx]

                if len(ts_idx) == 0:
                    continue    
                ts_idx = ts_idx[0]

                fake_all.append(regrEnergy)
                fake_pt_all.append(recoPt)
                
                print(assEv.ticlCandidate_recoToSim_CP, assEv.ticlCandidate_recoToSim_CP_sharedE, assEv.ticlCandidate_recoToSim_CP_score)
                recoToSim = assEv.ticlCandidate_recoToSim_CP[ts_idx]
                sharedE = assEv.ticlCandidate_recoToSim_CP_sharedE[ts_idx]
                score = assEv.ticlCandidate_recoToSim_CP_score[ts_idx]
                if not len(sharedE): continue
                argminScore = ak.argmin(score)
                if sharedE[argminScore]/regrEnergy < 0.1: continue
                simCand_idx = recoToSim[argminScore]
                print(ev, recoToSim, simCand_idx, cand_idx, ts_idx)

                if recoTrack != -1:
                    fake_with_track.append(regrEnergy)
                    fake_pt_with_track.append(recoPt)  
                
                print("before")
                simTrack = simCandEv.simTICLCandidate_track_in_candidate[simCand_idx]
                print("after")

                tk_idx_in_coll = -1
                try:
                    tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
                except:
                    continue
                if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                    continue
                   
                fake_denominator.append(regrEnergy)
                fake_pt_denominator.append(recoPt)
                
                if recoTrack != -1 and recoTrack != simTrack:
                    fake_numerator_strict.append(regrEnergy)
                    fake_pt_numerator_strict.append(recoPt)
                if recoTrack != simTrack:
                    fake_numerator.append(regrEnergy)
                    fake_pt_numerator.append(recoPt)                 
                                        
    return alls, with_track, efficients, denominator, numerator, pt_all, pt_with_track, pt_efficients, pt_denominator, pt_numerator, fake_all, fake_denominator, fake_with_track, fake_numerator, fake_numerator_strict, fake_pt_all, fake_pt_denominator, fake_pt_with_track, fake_pt_numerator, fake_pt_numerator_strict

alls, with_track, efficients, denominator, numerator, pt_all, pt_with_track, pt_efficients, pt_denominator, pt_numerator, fake_all, fake_denominator, fake_with_track, fake_numerator, fake_numerator_strict, fake_pt_all, fake_pt_denominator, fake_pt_with_track, fake_pt_numerator, fake_pt_numerator_strict = doEverything(dumperInputV5)
    
all_noTime, with_track_noTime, efficients_noTime, denominator_noTime, numerator_noTime, pt_all_noTime, pt_with_track_noTime, pt_efficients_noTime, pt_denominator_noTime, pt_numerator_noTime, fake_all_noTime, fake_denominator_noTime, fake_with_track_noTime, fake_numerator_noTime, fake_numerator_noTime_strict, fake_pt_all_noTime, fake_pt_denominator_noTime, fake_pt_with_track_noTime, fake_pt_numerator_noTime, fake_pt_numerator_noTime_strict = doEverything(dumperInputV5_noTime)

###############
## EFFICIENCY #
###############
#
#plt.figure()
#plt.hist(alls, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(with_track, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(efficients, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="efficient", hatch='/')
#plt.hist(denominator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'gold', label="denominator")
#plt.hist(numerator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator")
#plt.legend()
#plt.xlabel(("Sim Regressend Energy [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_regrEn.png")
#
#plt.figure()
#plt.hist(all_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(with_track_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(efficients_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="efficient", hatch='/')
#plt.hist(denominator_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'gold', label="denominator")
#plt.hist(numerator_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator")
#plt.legend()
#plt.xlabel(("Sim Regressend Energy [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_regrEn_noTime.png")
#
#plot_ratio_multiple([numerator, numerator_noTime], [denominator, denominator_noTime], 10, [1,200], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_vsEne_v5.png", ratio_label="time/no time")
#
#plt.figure()
#plt.hist(pt_all, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(pt_with_track, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(pt_efficients, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="efficient", hatch='/')
#plt.hist(pt_denominator, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'gold', label="denominator")
#plt.hist(pt_numerator, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator")
#plt.legend()
#plt.xlabel(("Sim Raw pT [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_rawPt.png")
#
#plt.figure()
#plt.hist(pt_all_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(pt_with_track_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(pt_efficients_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="efficient", hatch='/')
#plt.hist(pt_denominator_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'gold', label="denominator")
#plt.hist(pt_numerator_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator")
#plt.legend()
#plt.xlabel(("Sim Raw pT [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_rawPt_noTime.png")
#
#plot_ratio_multiple([pt_numerator, pt_numerator_noTime], [pt_denominator, pt_denominator_noTime], 10, [1,50], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Sim Raw Pt [GeV]", saveFileName=OutputDir + "trackEff_vsPt_v5.png", ratio_label="time/no time")
#
#########
## FAKE # fake_pt_all, fake_pt_denominator, fake_pt_noTrack, fake_pt_numerator
#########
#
#plt.figure()
#plt.hist(fake_all, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(fake_with_track, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(fake_denominator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="denominator")
#plt.hist(fake_numerator, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'gold', label="numerator")
#plt.hist(fake_numerator_strict, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator strict")
#plt.legend()
#plt.xlabel(("Raw Energy [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_fake_rawEn.png")
#
#plt.figure()
#plt.hist(fake_all_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(fake_with_track_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(fake_denominator_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="denominator")
#plt.hist(fake_numerator_noTime, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'gold', label="numerator")
#plt.hist(fake_numerator_noTime_strict, range=(1,200), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator strict")
#plt.legend()
#plt.xlabel(("Raw Energy [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_fake_rawEn_noTime.png")
#
#plot_ratio_multiple([numerator, numerator_noTime], [denominator, denominator_noTime], 10, [1,200], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_vsEne_v5.png", ratio_label="time/no time")
#plot_ratio_multiple([fake_numerator, fake_numerator_noTime], [fake_denominator, fake_denominator_noTime], 10, [1,200], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Raw Energy [GeV]", saveFileName=OutputDir + "trackFake_vsEne_v5.png", ratio_label="time/no time")
#plot_ratio_multiple([fake_numerator_strict, fake_numerator_noTime_strict], [fake_denominator, fake_denominator_noTime], 10, [1,200], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Raw Energy [GeV]", saveFileName=OutputDir + "trackFake_strict_vsEne_v5.png", ratio_label="time/no time")
#
#plt.figure()
#plt.hist(fake_pt_all, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(fake_pt_with_track, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(fake_pt_denominator, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="denominator")
#plt.hist(fake_pt_numerator, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'gold', label="numerator")
#plt.hist(fake_pt_numerator_strict, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator strict")
#plt.legend()
#plt.xlabel(("Raw pT [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_fake_rawPt.png")
#
#plt.figure()
#plt.hist(fake_pt_all_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'midnightblue', label="all")
#plt.hist(fake_pt_with_track_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'dodgerblue', label="with good track")
#plt.hist(fake_pt_denominator_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'forestgreen', label="denominator")
#plt.hist(fake_pt_numerator_noTime, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'gold', label="numerator")
#plt.hist(fake_pt_numerator_noTime_strict, range=(1,50), bins = 20, histtype = "step", lw = 2, color = 'orangered', label="numerator strict")
#plt.legend()
#plt.xlabel(("Raw pT [GeV]"))
#plt.savefig(OutputDir + "simTICLCand_fake_rawPt_noTime.png")
#
#plot_ratio_multiple([pt_numerator, pt_numerator_noTime], [pt_denominator, pt_denominator_noTime], 10, [1,50], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Sim Raw Pt [GeV]", saveFileName=OutputDir + "trackEff_vsPt_v5.png", ratio_label="time/no time")
#plot_ratio_multiple([fake_pt_numerator, fake_pt_numerator_noTime], [fake_pt_denominator, fake_pt_denominator_noTime], 10, [1,50], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Raw Pt [GeV]", saveFileName=OutputDir + "trackFake_vsPt_v5.png", ratio_label="time/no time")
#plot_ratio_multiple([fake_pt_numerator_strict, fake_pt_numerator_noTime_strict], [fake_pt_denominator, fake_pt_denominator_noTime], 10, [1,50], labels=["with time", "no time compatibility"], colors=['blue', 'red'], xlabel="Raw Pt [GeV]", saveFileName=OutputDir + "trackFake_strict_vsPt_v5.png", ratio_label="time/no time")
#
#
