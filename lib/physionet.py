###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lib.utils import get_device

# Adapted from time-series-datasets. Modified to preprocess local files only.

class PhysioNet(object):

	raw_files = ('set-a.tar.gz', 'set-b.tar.gz', 'set-c.tar.gz')

	params = [
		'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
		'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
		'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
		'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
	]

	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
	labels_dict = {k: i for i, k in enumerate(labels)}

	def __init__(self, root, download = False,
		quantization = None, n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.device = torch.device(device)
		self.reduce = "average"
		self.quantization = quantization

		if os.path.exists(os.path.join(self.root, self.set_a)) and \
		   os.path.exists(os.path.join(self.root, self.set_b)) and \
		   os.path.exists(os.path.join(self.root, self.set_c)):
			folder = self.root
		else:
			if download:
				self.download()

			if not self._check_exists():
				raise RuntimeError('Dataset not found. Supply the processed PhysioNet tensors or local archives in raw/.')
			folder = self.processed_folder

		data_a = torch.load(os.path.join(folder, self.set_a), map_location=self.device, weights_only=True)
		data_b = torch.load(os.path.join(folder, self.set_b), map_location=self.device, weights_only=True)
		data_c = torch.load(os.path.join(folder, self.set_c), map_location=self.device, weights_only=True)

		self.data = data_a + data_b + data_c # a list with length 12000

		if n_samples is not None:
			print('Total records:', len(self.data))
			self.data = self.data[:n_samples]

	def download(self):
		"""Preprocess local archives; retain the upstream method name for compatibility."""
		if self._check_exists():
			return
		for filename in self.raw_files:
			if not os.path.isfile(os.path.join(self.raw_folder, filename)):
				raise FileNotFoundError(f'Missing local PhysioNet data: {os.path.join(self.raw_folder, filename)}. See README.md.')

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		black_list = [140501, 150649, 140936, 143656, 141264, 145611, 142998, 147514, 142731,150309, 155655, 156254]
		for filename in self.raw_files:
			tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
			tar.extractall(self.raw_folder)
			tar.close()

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder, filename.split('.')[0])
			patients = []
			total = 0
			cnt = 0
			for txtfile in os.listdir(dirname):
				record_id = txtfile.split('.')[0]
				if record_id in black_list:
					continue
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = 0
					tt = [0.]
					vals = [torch.zeros(len(self.params))]
					mask = [torch.zeros(len(self.params))]
					nobs = [torch.zeros(len(self.params))]
					for l in lines[1:]:
						total += 1
						time, param, val = l.split(',')
						# Time in hours
						time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.

						# round up the time stamps (up to 6 min by default)
						# used for speed -- we actually don't need to quantize it in Latent ODE
						if(self.quantization != None and self.quantization != 0):
							time = round(time / self.quantization) * self.quantization

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.params)))
							mask.append(torch.zeros(len(self.params)))
							nobs.append(torch.zeros(len(self.params)))
							prev_time = time

						if param in self.params_dict:
							n_observations = nobs[-1][self.params_dict[param]]
							if self.reduce == 'average' and n_observations > 0:
								prev_val = vals[-1][self.params_dict[param]]
								new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
								vals[-1][self.params_dict[param]] = new_val
							else:
								vals[-1][self.params_dict[param]] = float(val)
							mask[-1][self.params_dict[param]] = 1
							nobs[-1][self.params_dict[param]] += 1
						else:
							assert (param == 'RecordID' or param ==''), 'Read unexpected param {}'.format(param)
							if(param != 'RecordID'):
								cnt += 1
								print(cnt, param, l)

				tt = torch.tensor(tt).to(self.device)
				vals = torch.stack(vals).to(self.device)
				mask = torch.stack(mask).to(self.device)

				patients.append((record_id, tt, vals, mask))

			torch.save(
				patients,
				os.path.join(self.processed_folder,
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			)

		print('Done!')

	def _check_exists(self):
		for filename in self.raw_files:

			if not os.path.exists(
				os.path.join(self.processed_folder,
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def set_a(self):
		return 'set-a_{}.pt'.format(self.quantization)

	@property
	def set_b(self):
		return 'set-b_{}.pt'.format(self.quantization)

	@property
	def set_c(self):
		return 'set-c_{}.pt'.format(self.quantization)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Quantization: {}\n'.format(self.quantization)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str

	def visualize(self, timesteps, data, mask, plot_name):
		width = 15
		height = 15

		non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		mask = mask[:, non_zero_idx]
		data = data[:, non_zero_idx]

		params_non_zero = [self.params[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		n_col = 3
		n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
		fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		#for i in range(len(self.params)):
		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			tp_mask = mask[:,param_id].long()

			tp_cur_param = timesteps[tp_mask == 1.]
			data_cur_param = data[tp_mask == 1., param_id]

			ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o')
			ax_list[i // n_col, i % n_col].set_title(param)

		fig.tight_layout()
		fig.savefig(plot_name)
		plt.close(fig)

def task_mask(args, total_dataset):
	total_dataset_new = []
	for n, (record_id, tt, vals, mask) in enumerate(total_dataset):
		if(args.task == 'forecasting'):
			mask_observed_tp = torch.lt(tt, args.history)
		elif(args.task == 'imputation'):
			mask_observed_tp = torch.ones_like(tt).bool()
			rng = np.random.default_rng(n)
			mask_inds = rng.choice(len(tt), size=int(len(tt)*args.mask_rate), replace=False)
			# mask_inds = np.random.choice(len(tt), size=int(len(tt)*args.mask_rate), replace=False)
			# print(n,'-', mask_inds, mask_inds.sum())
			mask_observed_tp[mask_inds] = False
			if args.dataset in ["physionet", "mimic"]:
				mask_observed_tp[0] = True # in some datasets, only time 0 has values
			# print(len(tt), mask_observed_tp.sum())
		else:
			raise Exception('{}: Wrong task specified!'.format(args.task))
		total_dataset_new.append((record_id, tt, vals, mask, mask_observed_tp))

	return total_dataset_new


def get_data_min_max(records, device):
	inf = torch.Tensor([float("Inf")])[0].to(device)

	data_min, data_max, time_max = None, None, -inf

	for b, record_tuple in enumerate(records):
		record_id, tt, vals, mask = record_tuple[0], record_tuple[1], record_tuple[2], record_tuple[3]
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

		time_max = torch.max(time_max, tt.max())

	missing_feature_mask = torch.isinf(data_min) | torch.isinf(data_max)
	if missing_feature_mask.any():
		print('features without observed training values:', torch.where(missing_feature_mask)[0].detach().cpu().tolist())
		data_min = data_min.clone()
		data_max = data_max.clone()
		data_min[missing_feature_mask] = 0.
		data_max[missing_feature_mask] = 1.

	print('data_max:', data_max)
	print('data_min:', data_min)
	print('time_max:', time_max)

	return data_min, data_max, time_max

def get_seq_length(args, records):

	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask, mask_observed) in enumerate(records):
		n_observed_tp = mask_observed.sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len

def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train",
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	observed_tp = []
	observed_data = []
	observed_mask = []
	predicted_tp = []
	predicted_data = []
	predicted_mask = []

	for b, (record_id, tt, vals, mask, mask_observed) in enumerate(batch):
		observed_tp.append(tt[mask_observed])
		observed_data.append(vals[mask_observed])
		observed_mask.append(mask[mask_observed])

		mask_predicted = ~mask_observed
		predicted_tp.append(tt[mask_predicted])
		predicted_data.append(vals[mask_predicted])
		predicted_mask.append(mask[mask_predicted])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	if(args.dataset != 'ushcn'):
		observed_data = utils.normalize_masked_data(observed_data, observed_mask,
			att_min = data_min, att_max = data_max)
		predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask,
			att_min = data_min, att_max = data_max)

	observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)

	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}

	return data_dict

def variable_time_collate_series(batch, args, device, return_np=False, to_set=False, maxlen=None,
                            data_min=None, data_max=None, time_max=None, classify=False, activity=False):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L, D) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	# n_observed_tps = []
	observed_tp = []
	observed_data = []
	observed_mask = []
	predicted_tp = []
	predicted_data = []
	predicted_mask = []
	# print(batch[0].shape)
	D = batch[0][2].shape[1]
	if maxlen is None:
		if activity == False:
			seq_lens = [ex[3].sum(dim=0).max().item() for ex in batch]
			maxlen = int(np.max(seq_lens))
		else:
			seq_lens = [ex[1].size(0) for ex in batch]
			maxlen = int(np.max(seq_lens))

	for b, (record_id, tt, vals, mask, mask_observed) in enumerate(batch):
		observed_tp.append(tt[mask_observed])
		observed_data.append(vals[mask_observed])
		observed_mask.append(mask[mask_observed])

		mask_predicted = ~mask_observed
		predicted_tp.append(tt[mask_predicted])
		predicted_data.append(vals[mask_predicted])
		predicted_mask.append(mask[mask_predicted])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	observed_len = observed_tp.shape[1]
	observed_combined_tt = torch.zeros([len(batch), observed_len, D]).to(device)
	observed_combined_vals = torch.zeros([len(batch), observed_len, D]).to(device)
	observed_combined_mask = torch.zeros([len(batch), observed_len, D]).to(device)

	# mx_len = 0
	b = 0
	for tt,vals,mask in zip(observed_tp, observed_data, observed_mask):
		for d in range(D):
			mask_bd = mask[:,d].bool()
			currlen = int(mask_bd.sum())
			observed_combined_tt[b, :currlen, d] = tt[mask_bd]
			observed_combined_vals[b, :currlen, d] = vals[mask_bd,d]
			observed_combined_mask[b, :currlen, d] = mask[mask_bd,d]
		b += 1

	observed_data = utils.normalize_masked_data(observed_combined_vals, observed_combined_mask,
			att_min = data_min, att_max = data_max)
	predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask,
			att_min = data_min, att_max = data_max)

	observed_tp = utils.normalize_masked_tp(observed_combined_tt, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)

	# print(observed_combined_mask[0])

	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_combined_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	return data_dict

if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = PhysioNet('../data/physionet', train=False, download=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	print(dataloader.__iter__().next())
