import time
from typing import List
from torch.utils.data import Dataset
from load_dataset.sleepagent import SleepAgent
from utils.finder import walk
import numpy as np
import os, sys
import mne
class Dataset_SleepEdf(SleepAgent):
  """
  load raw data from sleepedf
  """
  def __init__(self, root_path, flag='train',
               data_path='sleepedf', channels=[0], subsets=None):
    super(Dataset_SleepEdf).__init__()
    self.root_path = os.path.join(root_path,data_path)
    self.flag = flag
    self.dataset_name = data_path
    if subsets == None: self.subsets = [[0, 1], [2, 3], [3, 4]]

    self.load_data(channels=channels)


  def load_data(self, channels, rewrite=False):
    if not isinstance(channels, list): channels = list(channels)
    # Traverse all hypnogram files
    hypnogram_file_list: List[str] = walk(self.root_path, 'file', '*Hypnogram*',
                                       return_basename=True)
    n_patients = len(hypnogram_file_list)
    assert self.flag in ['train', 'val', 'test']
    if self.flag == 'train':
      self.subsets = self.subsets[0]
    elif self.flag == 'val':
      self.subsets = self.subsets[1]
    else:
      self.subsets = self.subsets[2]
    hypnogram_file_list = [hypnogram_file_list[subset] for subset in self.subsets]

    data_groups = []
    label_groups = []

    for i, hypnogram_file in enumerate(hypnogram_file_list):
      # Get id
      id: str = os.path.split(hypnogram_file)[-1].split('-')[0][:7]
      npz_path = os.path.join(self.root_path, id + '(raw)' + '.npz')

      if os.path.exists(npz_path) and not rewrite:
        # 直接读取npz文件
        edf_data, annotations = self._load_npz_data(npz_path, rewrite=rewrite)

      else:
        # read the annotations
        hypnogram_file = os.path.join(self.root_path, hypnogram_file)
        raw_annotations = self.read_annotations_mne(hypnogram_file)

        # read the edf
        fn = os.path.join(self.root_path, id + '0' + '-PSG.edf')
        raw_edf_data = self.read_edf_mne(fn)

        edf_data, annotations = self.preprocess_raw_edf(raw_edf_data, raw_annotations)
        self._save_npz_data(npz_path, edf_data, annotations, rewrite=rewrite)

      # process the data

      data_groups.append(edf_data[:, :, channels])
      label_groups.append(annotations)

    # return data_groups, label_groups
    self.x = np.concatenate(data_groups, axis=0)
    self.y = np.concatenate(label_groups, axis=0)

  def __getitem__(self, item):
    return self.x[item], self.y[item]

  def __len__(self):
    return self.x.shape[0]

if __name__ == '__main__':
  data_path = r'E:\eason\project\dp_pytorch\data'
  sleepedf = Dataset_SleepEdf(root_path=data_path)
  pass



# class Dataset_UcddbEdf(Dataset):
#   """
#   load raw data from sleepedf
#   """
#   print('duqu edf')
#
# class Dataset_RRSHEdf(Dataset):
#   """
#   load raw data from sleepedf
#   """
#   print('duqu edf')