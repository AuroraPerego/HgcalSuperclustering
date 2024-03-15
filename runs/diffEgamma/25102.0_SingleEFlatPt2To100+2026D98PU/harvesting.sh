
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/CMSSW_14_0_0_pre0
cmsenv
cd - 

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-dnn
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98PU/step4_HARVESTING_PU_local.py

cd /data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-mustache
cmsRun /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs/runs/diffEgamma/25102.0_SingleEFlatPt2To100+2026D98PU/step4_HARVESTING_PU_local.py


# COmapring two DQM files for differences
(
    cd /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU
    mkdir RelMonCompare
    mkdir RelMonCompareEgamma
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompare -CR --standalone --use_black_file -p --no_successes &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d Egamma &
compare_using_files.py /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-mustache/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root /grid_mnt/data_cms_upgrade/cuisset/supercls/diffEgamma/25102_SingleE_D98PU/v2-5362bd-dnn/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -o RelMonCompareEgamma -CR --standalone --use_black_file -p -d EgammaV &
)