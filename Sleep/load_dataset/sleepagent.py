import torch
from torch.utils.data import Dataset, DataLoader
import mne
import numpy as np
from typing import Dict
import os
class SleepAgent(Dataset):

  ANNO_KEY = 'stage Ground-Truth'
  EPOCH_DURATION = 30.0
  CHANNELS = {}

  ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
  }

  @classmethod
  def read_annotations_mne(cls, data_dir) -> Dict[str, np.ndarray]:
    mne_anno: mne.Annotations = mne.read_annotations(data_dir)

    return mne_anno
    # raise NotImplementedError
  @classmethod
  def read_edf_mne(cls, data_dir):
    raw = mne.io.read_raw_edf(data_dir, preload=False, verbose=False)
    sampling_rate = raw.info['sfreq']
    signal_dict = {}
    if sampling_rate not in signal_dict: signal_dict[sampling_rate] = []
    signal_dict[sampling_rate].append((raw.ch_names, raw.get_data()))
    return signal_dict
    pass

  @classmethod
  def preprocess_raw_edf(cls, signal_dict, annotations):
    print('processs the raw')
    for sfreq, signal_lists in signal_dict.items():
      data = np.concatenate([x for _, x in signal_lists], axis=0)
      channel_names = [name for names, _ in signal_lists for name in names]
    # 将原label转成 0, 1, 2, 3, 4, 5
    interval, labels = [], []

    for onset, duration, label in zip(annotations.onset, annotations.duration, annotations.description):
      interval.append((onset, onset + duration))
      labels.append(cls.ann2label[label])

    assert len(labels) == len(interval)

     # labels interval 为list， 其他的为numpy array
    if labels[0] == 0:
      labels.pop(0)
      interval.pop(0)
      annotations.duration = annotations.duration[1:]
      annotations.onset = annotations.onset[1:]
    data_interval = np.array(interval).astype(int) * 100
    cls.labels = [labels[0]] * int(annotations.duration[0] / 30)
    cls.data = data[:, data_interval[0][0]:data_interval[0][1]]
    for i in range(1, len(labels)):
      if labels[i] == 5:
        print('tichu yici 5')
      else:
        cls.labels += [labels[i]] * int(annotations.duration[i] / 30)
        cls.data = np.concatenate((cls.data, data[:, data_interval[i][0]:data_interval[i][1]]), axis=1)

    return cls.data, np.array(cls.labels)
  def _load_npz_data(self,npz_path, rewrite =True):

    if not rewrite and os.path.exists(npz_path):
      dg = np.load(npz_path, allow_pickle=True)
      edf_data, annotations = dg['data'], dg['label']
      edf_data = edf_data.T
      edf_data = edf_data.reshape(-1, 3000, edf_data.shape[1])
      print('have loaded {}'.format(npz_path))
      return edf_data, annotations
  def _save_npz_data(self, npz_path, edf_data, anotations, rewrite= False):
    if rewrite or not os.path.exists(npz_path):
      # if anotations
      save_dict = {
        'data': edf_data,
        'label': anotations
      }
      np.savez(npz_path, **save_dict)
      print('save the {}'.format(npz_path))