# TemporalRegularity_RWD2023
 Repo for code for DiTullio et al 2023 Frontiers Manuscript

SFA tools contains primary code for running the analyses of the paper.

SFA_singleexample provides an illustrative example of how the algorithm works.

single_runSFA allows the user to explore different simulation situations and calculations with some example vocalization pairs.

Each other name_runSFA file is looping code for a different result of the paper.

Basic Stats and Improved_localplotting will generate figures and the statistics used in the paper based on the data in the Results folder.

Example folder contains example stimuli for each stimulus category that can be used by single_runSFA or the other name_runSFA files by creating a new pair list.

MATLAB code file contains the code used to generate the modulation spectra.  A separate readme file for this code will be generated soon.

Required modules:

numpy
pandas
seaborn
pathlib
scipy
matplotlib
pyfilterbank (included in repo)
sklearn 
tqdm ( found here: https://github.com/tqdm/tqdm)
soundfile (found here: https://pypi.org/project/soundfile/ )
