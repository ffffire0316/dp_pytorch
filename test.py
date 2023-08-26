import numpy as np
import mne
data_2d = np.random.rand(3 * 3000, 7)

data = data_2d.reshape(3, 3000, data_2d.shape)