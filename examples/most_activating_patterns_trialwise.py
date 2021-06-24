"""
Most Activating Input Patterns
======================================

"""




######################################################################
# In Braindecode, there are two supported configurations created for
# training models: trialwise decoding and cropped decoding. We will
# explain this visually by comparing trialwise to cropped decoding.
#
# .. image:: ../_static/trialwise_explanation.png
# .. image:: ../_static/cropped_explanation.png
#
# On the left, you see trialwise decoding:
#
# 1. A complete trial is pushed through the network.
# 2. The network produces a prediction.
# 3. The prediction is compared to the target (label) for that trial to
#    compute the loss.
#
# On the right, you see cropped decoding:
#
# 1. Instead of a complete trial, crops are pushed through the network.
# 2. For computational efficiency, multiple neighbouring crops are pushed
#    through the network simultaneously (these neighbouring crops are
#    called compute windows)
# 3. Therefore, the network produces multiple predictions (one per crop in
#    the window)
# 4. The individual crop predictions are averaged before computing the
#    loss function
#
# .. note::
#
#     -  The network architecture implicitly defines the crop size (it is the
#        receptive field size, i.e., the number of timesteps the network uses
#        to make a single prediction)
#     -  The window size is a user-defined hyperparameter, called
#        ``input_window_samples`` in Braindecode. It mostly affects runtime
#        (larger window sizes should be faster). As a rule of thumb, you can
#        set it to two times the crop size.
#     -  Crop size and window size together define how many predictions the
#        network makes per window: ``#window−#crop+1=#predictions``
#


######################################################################
# .. note::
#     For cropped decoding, the above training setup is mathematically
#     identical to sampling crops in your dataset, pushing them through the
#     network and training directly on the individual crops. At the same time,
#     the above training setup is much faster as it avoids redundant
#     computations by using dilated convolutions, see our paper
#     `Deep learning with convolutional neural networks for EEG decoding and visualization <https://arxiv.org/abs/1703.05051>`_.  # noqa: E501
#     However, the two setups are only mathematically identical in case (1)
#     your network does not use any padding or only left padding and
#     (2) your loss function leads
#     to the same gradients when using the averaged output. The first is true
#     for our shallow and deep ConvNet models and the second is true for the
#     log-softmax outputs and negative log likelihood loss that is typically
#     used for classification in PyTorch.
#





######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#

import mne

from braindecode.datasets.moabb import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, preprocess, Preprocessor)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


######################################################################
# Create model and compute windowing parameters
# ---------------------------------------------
#
import numpy as np
from braindecode.datautil.windowers import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)



######################################################################
# Splitting the data into training and validation set
#
#
#

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

###################################################################
# Creating a model


import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Send model to GPU
if cuda:
    model.cuda()





######################################################################
#  Training the model
#
#
#
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

# These values we found good for shallow network:

# For deep4 they should be:
lr = 1 * 0.01
weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 100   #this we should change to 8 epochs afterwards

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

######################################################################


######################################################################
# getting the model

model = clf.module

######################################################################
# Let's construct a submodel for the layer we are interested in
#
from torch import nn

wanted_layer = 'conv_2'

submodel = nn.Sequential()
for name, module in model.named_children():
    submodel.add_module(name, module)
    if name == wanted_layer:
        break
print(submodel)


######################################################################
# Get all activations for this layer
#
trainloader = torch.utils.data.DataLoader(
    train_set,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    batch_size=64)


# Could do on train set/valid set...
# Here we do on train set
# In future, should improve braindecode to have function
# Get non-overlapping activations (train_set) (but is that possible? maybe just assume defaults)
#here I chould add check for device!

all_X = []
all_acts = []
with torch.no_grad():
    for X,y,i in trainloader:
        all_X.append(X)
        #act = submodel(X.cuda())
        act = submodel(X)
        print(np.shape(act))
        #all_acts.append(act.cpu().detach().numpy())
        all_acts.append(act.numpy())
all_X = np.concatenate(all_X)
all_acts = np.concatenate(all_acts)
print(all_X.shape) #(384*22*1000)  trials*channels*timepoints
print(all_acts.shape)  #(384*50*962*1)


######################################################################
# calculate how many input samples the network needs to produce one
#activation (receptive field size)
#formula for cropped case is incorrect here, just hardcoded the RF size for now
#
n_receptive_field = all_X.shape[2] - all_acts.shape[2] + 1
n_receptive_field = 39

######################################################################
# Decide which unit to investigate
#

i_unit = 0
unit_acts = all_acts[:,i_unit].squeeze()
unit_acts.shape  #trial*unit_output 288*363


######################################################################
# Sort activations
# (we sort absolute activations, but could also sort raw activations

# sort 2d array, see https://stackoverflow.com/a/64338853/1469195
i_acts_sorted = np.stack(np.unravel_index(np.argsort(np.abs(unit_acts), axis=None), unit_acts.shape),axis=1)
n_fields_to_collect = 100
all_most_activating_windows = []
for i_batch, i_act in i_acts_sorted[-n_fields_to_collect:]:
    # Start of receptive field
    i_start = i_act
    # One past end of receptive field
    i_stop = i_act + n_receptive_field
    X_part = all_X[i_batch,:,i_start:i_stop]
    print(np.shape(X_part), i_start, i_stop)
    all_most_activating_windows.append(X_part)
all_most_activating_windows = np.array(all_most_activating_windows, dtype = object)


#lowest absolute activation



#highest absolute activation


unit_acts[i_acts_sorted[-1][0], i_acts_sorted[-1][1]]

######################################################################
# Visualize most activating receptive fields
#
import numpy as np
import matplotlib as mpl
mpl.use("module://backend_interagg") #this is necessary to display plots in SciView in Pycharm
import matplotlib.pyplot as plt


######################################################################
# Let's plot sorted activations in different colors
#

acts_sorted = np.array([unit_acts[i_b,i_t] for i_b,i_t in i_acts_sorted])
plt.plot(acts_sorted[acts_sorted>0], '.', color = 'red', label = 'positive');
plt.plot(acts_sorted[acts_sorted<0], '.', color = 'blue', label = 'negative')
#plt.plot(acts_sorted, '.')
plt.xlabel('Sorted Activation Index')
plt.ylabel('Activation Value')
plt.legend()
plt.grid(True)
plt.show()

####get montage
montage = dataset.datasets[0].raw.get_montage()
#get channel list
ch_names = montage.ch_names
dig = montage.dig


tight_cap_positions = [
['', '', '', 'Fz', '', '', ''],
['', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', ''],
[ 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
['', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', ''],
['', '', 'P1', 'Pz', 'P2', '', ''],
['', '', '', 'POz', '', '', '']]

plt.plot(all_most_activating_windows[:,0].T);
plt.xlabel('timesteps')
plt.ylabel('normalized EEG')

def get_sensor_pos(sensor_name, sensor_map=tight_cap_positions):
    sensor_pos = np.where(
        np.char.lower(np.char.array(sensor_map)) == sensor_name.lower())
    # unpack them: they are 1-dimensional arrays before
    assert len(sensor_pos[0]) == 1, (
        "there should be a position for the sensor "
        "{:s}".format(sensor_name))
    return sensor_pos[0][0], sensor_pos[1][0]

def plot_head_signals_tight(signals, sensor_names=None, figsize=(12, 7),
                            plot_args=None, hspace=0.35,
                            sensor_map=tight_cap_positions,
                            tsplot=False, sharex=True, sharey=True):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
                                                                       "sensor names for all sensor matrices")
    assert sensor_names is not None
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in
                        sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    for i in range(0, len(signals)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (
                                  row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        elif sharex is True and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax,
                                    sharex=first_ax)
        elif sharex is True and sharey is False:
            ax = figure.add_subplot(rows, cols, subplot_ind,
                                    sharex=first_ax)
        elif sharex is False and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax)
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind)

        signal = signals[i]
        if tsplot is False:
            ax.plot(signal, **plot_args)
        else:
            seaborn.tsplot(signal.T, ax=ax, **plot_args)
        ax.set_title(sensor_name)
        ax.set_yticks([])
        if len(signal) == 600:
            ax.set_xticks([150, 300, 450])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([])

        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0, ls=':', color="grey")
        figure.subplots_adjust(hspace=hspace)
    return figure



fig = plot_head_signals_tight(np.median(all_most_activating_windows, axis=0),
                       sensor_names=ch_names, figsize=(16,16))


#here i am trying to plot the tight-layout with MNE function...

n_channels = 22
sampling_freq = 250.
info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq = sampling_freq)
print(info)
info['description'] = 'Maximally activating windows'
activations_epochs = mne.EpochsArray(all_most_activating_windows, info=info)
activations_epochs.set_montage(montage)
evokeds = [activations_epochs.average()]
mne.viz.plot_evoked_topo(evokeds)


activations_epochs.info


#another MNE version of the plot, just smth is wrong with smoothing
from mne.viz import iter_topography

for ax, idx in iter_topography(dataset.datasets[0].raw.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white'):
    ax.plot(np.mean(all_most_activating_windows, axis = 0)[idx], color='blue')
    ax.set_title(ch_names[idx])
    plt.grid(True)

plt.gcf().suptitle('Activation Windows')
plt.show()