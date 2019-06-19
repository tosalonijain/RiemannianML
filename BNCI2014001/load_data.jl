""" This script shows how to load the data from the BNCI2014001 dataset where
subject are performing a task of left/right hand motor imagery (see:
http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html#moabb.datasets.BNCI2014001)

This dataset contains the recording of 9 sujbects during two different sessions.
Each is stored in a different file and has a different subject number.
1 = subject 1, session 1
2 = subject 1, session 2
3 = subject 2, session 1
4 = subject 2, session 2
...
17 = subject 9, session 1
18 = subject 9, session 2

Each file contains an array X of the EEG recording (ntrials x nchannels x nsamples)
and array y collecting the corresponding labels (1 = left hand; 2 = right hand)

The channels has been rearranged according to the following order:
["FC3", "FC1", "C1", "C3", "C5", "CP3", "CP1", "P1", "POZ", "PZ", "CPZ", "FZ",
"FC4", "FC2", "FCZ", "CZ", "C2", "C4", "C6", "CP4", "CP2", "P2"] """


# To load the data in Julia
using NPZ

path = "" # where files are stored
filename = "subject_i.npz" # for subject number i
data = npzread(path*filename)
X = data["data"] # retrive the epochs
y = data["labels"] # retrive the corresponding labels


# To load the data in Python
import numpy as np

path = "" # where files are stored
filename = "subject_i.npz" # for subject number i
data = np.load(path + filename)
X = data["data"] # retrive the epochs
y = data["labels"] # retrive the corresponding labels
