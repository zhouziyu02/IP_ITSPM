import os
import sys
from pathlib import Path

import time
import argparse
import numpy as np
import pandas as pd
import datetime
from random import SystemRandom
from models.ITSPM import ITSPM


import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils

from lib.parse_datasets import parse_datasets
from lib.evaluation import *

import warnings
warnings.filterwarnings('ignore')


class ForecastingDataParallelWrapper(nn.Module):
	def __init__(self, backbone):
		super().__init__()
		self.backbone = backbone

	def forward(self, tp_to_predict, observed_data, observed_tp, observed_mask=None):
		pred_y = self.backbone.forecasting(tp_to_predict, observed_data, observed_tp, observed_mask)
		if pred_y.dim() == 4 and pred_y.size(0) == 1:
			return pred_y.squeeze(0).contiguous()
		return pred_y.contiguous()

	def forecasting(self, tp_to_predict, observed_data, observed_tp, observed_mask=None):
		return self.forward(tp_to_predict, observed_data, observed_tp, observed_mask)


class BatchOnlyDataParallel(nn.DataParallel):
	def scatter(self, inputs, kwargs, device_ids):
		if len(inputs) != 4:
			return super().scatter(inputs, kwargs, device_ids)

		tp_to_predict, observed_data, observed_tp, observed_mask = inputs
		n_devices = min(len(device_ids), observed_data.size(0))
		device_ids = device_ids[:n_devices]

		data_chunks = observed_data.tensor_split(n_devices, dim=0)
		tp_chunks = observed_tp.tensor_split(n_devices, dim=0) if observed_tp.dim() > 1 and observed_tp.size(0) == observed_data.size(0) else None
		mask_chunks = None
		if observed_mask is not None and observed_mask.dim() > 1 and observed_mask.size(0) == observed_data.size(0):
			mask_chunks = observed_mask.tensor_split(n_devices, dim=0)

		scattered_inputs = []
		for idx, device_id in enumerate(device_ids):
			device = torch.device("cuda", device_id)
			local_tp_to_predict = tp_to_predict
			if tp_to_predict.dim() > 1 and tp_to_predict.size(0) == observed_data.size(0):
				local_tp_to_predict = tp_to_predict.tensor_split(n_devices, dim=0)[idx]

			local_observed_tp = tp_chunks[idx] if tp_chunks is not None else observed_tp
			local_observed_mask = mask_chunks[idx] if mask_chunks is not None else observed_mask
			if local_observed_mask is not None:
				local_observed_mask = local_observed_mask.to(device, non_blocking=True)

			scattered_inputs.append((
				local_tp_to_predict.contiguous().to(device, non_blocking=True),
				data_chunks[idx].contiguous().to(device, non_blocking=True),
				local_observed_tp.contiguous().to(device, non_blocking=True),
				local_observed_mask.contiguous() if local_observed_mask is not None else None,
			))

		return tuple(scattered_inputs), tuple({} for _ in scattered_inputs)


def get_regression_eval_model(model):
	if isinstance(model, BatchOnlyDataParallel):
		return model.module.backbone
	return model


def load_regression_checkpoint(path, model):
	if not os.path.isfile(path):
		raise FileNotFoundError(f'Regression checkpoint not found: {path}')
	checkpoint = torch.load(path, map_location='cpu', weights_only=True)
	get_regression_eval_model(model).load_state_dict(checkpoint['model_state_dict'])
	return checkpoint


def save_regression_checkpoint(path, model, epoch):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	torch.save({'model_state_dict': get_regression_eval_model(model).state_dict(),
	            'epoch': epoch}, path)

parser = argparse.ArgumentParser('ITS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('--model', type=str, default='ITSPM', help='model name (ITSPM)')

parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default=str(Path(__file__).resolve().parent / 'data'))

parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_types', type=int, default=23)

parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--log', type=str, default='./logs/')
parser.add_argument('--save_path', type=str, default='./checkpoints/regression/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--load_path', type=str, default=None, help='Path to a saved regression checkpoint.')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--task', type=str, required=True, choices=['imputation', 'forecasting'])

parser.add_argument('--debug_flag', action='store_true')
parser.add_argument('--dp_flag', action='store_true')
parser.add_argument('--load_in_batch', action='store_true')
parser.add_argument('--history', type=int, default=24, help="number of hours (or months for ushcn) as historical window")
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--median_len', type=int, default=50)
parser.add_argument('--load', type=str, default=None, help='Experiment ID for logs and checkpoint filenames; use --load_path to restore a model.')
parser.add_argument('--dataset', type=str, default='physionet', choices=['physionet', 'mimic', 'ushcn', 'activity'])
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")

parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--sample_rate', type=float, default=1.0)
parser.add_argument('--mask_rate', type=float, default=0.3)
parser.add_argument('--collate', type=str, default='indseq')

# ITSPM arguments
parser.add_argument('--n_ref_points', type=int, default=32)
parser.add_argument('--n_scales', type=int, default=3)
parser.add_argument('--n_mixer_layers', type=int, default=2)
parser.add_argument('--max_event_tokens', type=int, default=None)
parser.add_argument('--max_gap_tokens', type=int, default=None)

file_name = os.path.basename(__file__)[:-3]

#####################################################################################################


if __name__ == '__main__':

	args = parser.parse_args()
	if args.epoch <= 0 or args.batch_size <= 0 or args.history <= 0:
		parser.error('--epoch, --batch_size, and --history must be positive.')
	if not 0 < args.mask_rate < 1:
		parser.error('--mask_rate must be in (0, 1).')
	if args.test_only and args.load_path is None:
		parser.error('--test_only requires --load_path.')
	if args.load_path is not None and not os.path.isfile(args.load_path):
		parser.error(f'Checkpoint not found: {args.load_path}')
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.PID = os.getpid()
	print("PID, device:", args.PID, args.device)
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)

	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	default_log_dirs = {"imputation": "logs/interpolation", "forecasting": "logs/extrapolation"}
	log_dir = args.log
	if log_dir in ("./logs/", "logs/", "./logs", "logs"):
		log_dir = default_log_dirs.get(args.task, "logs")
	if(args.n < 12000):
		args.state = "debug"
		log_path = os.path.join(log_dir, f"{args.task}_{args.dataset}_{args.model}_{args.state}.log")
	else:
		log_path = os.path.join(
			log_dir,
			f"{args.task}_{args.dataset}_{args.model}_{args.state}_history{args.history}_"
			f"d{args.d_model}_ref{args.n_ref_points}_scale{args.n_scales}_lr{args.lr}.log",
		)

	if not os.path.exists(log_dir):
		utils.makedirs(log_dir)

	is_itspm = args.model.lower() in ['itspm']
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode, saving=not is_itspm)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	##################################################################
	data_obj = parse_datasets(args, length_stat=True)
	args.enc_in = data_obj["input_dim"]
	args.num_types =  data_obj["input_dim"]
	args.input_dim =  data_obj["input_dim"]
	args.median_len = data_obj["median_len"]
	args.input_len = data_obj["max_input_len"]
	args.pred_len = data_obj["max_pred_len"]

	### Model Config ###
	if args.model.lower() in ['itspm']:
		model = ITSPM(args).to(args.device)
	else:
		raise ValueError(f"Unsupported model '{args.model}'. This cleaned project keeps ITSPM only.")

	if args.dp_flag:
		if args.dataset.lower() == 'mimic' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
			model = BatchOnlyDataParallel(ForecastingDataParallelWrapper(model))
			print(f"[Info] Enabled MIMIC DataParallel on {torch.cuda.device_count()} visible GPUs: {args.gpu}")
		else:
			print("[Info] --dp_flag ignored: multi-GPU regression is only enabled for MIMIC with >1 visible CUDA device.")

	loaded_checkpoint = load_regression_checkpoint(args.load_path, model) if args.load_path else None
	checkpoint_path = os.path.join(args.root_path, args.save_path,
	                               f'{args.task}_{args.dataset}_{args.model}_{args.state}_{experimentID}.pt')

	### Optimizer ###
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	def save_regression_results(args, best_iter, test_res, training_duration):
		if test_res is None:
			return
		default_res_filename = 'Interpolation/results/Interpolation_results.txt' if args.task == 'imputation' else 'Extrapolation/results/Extrapolation_results.txt'
		res_filename = os.environ.get("ITSPM_REGRESSION_RESULTS_FILE", default_res_filename)
		task_name = 'Interpolation' if args.task == 'imputation' else 'Extrapolation'

		# Construct hyperparameter string
		hparams = []
		if args.model.lower() in ['itspm']:
			token_hparams = ""
			if args.max_event_tokens is not None or args.max_gap_tokens is not None:
				token_hparams = f", max_event_tokens: {args.max_event_tokens}, max_gap_tokens: {args.max_gap_tokens}"
			hparams.append(f"d_model: {args.d_model}, n_ref_points: {args.n_ref_points}, n_scales: {args.n_scales}, n_mixer_layers: {args.n_mixer_layers}{token_hparams}")
		hparams_str = ", ".join(hparams)

		import datetime
		from datetime import timezone, timedelta
		tz_utc_8 = timezone(timedelta(hours=8))
		time_now_str = datetime.datetime.now(tz_utc_8).strftime("%Y-%m-%d %H:%M:%S")

		minutes = int(training_duration // 60)
		seconds = int(training_duration % 60)
		training_time_str = f"{minutes}m {seconds}s"

		res_dir = os.path.dirname(res_filename)
		if res_dir:
			os.makedirs(res_dir, exist_ok=True)
		with open(res_filename, "a") as res_f:
			res_f.write(f"Model: {args.model}, Task: {task_name}, Dataset: {args.dataset}, Best Epoch: {best_iter}, Seed: {args.seed}, lr: {args.lr}, batch_size: {args.batch_size}, {hparams_str}\n")
			res_f.write(f"MSE: {test_res['mse']:.5f}, MAE: {test_res['mae']:.5f}\n")
			res_f.write(f"Time now: {time_now_str}, Time for training: {training_time_str}\n\n")

	best_val_mse = np.inf
	best_iter = -1
	test_res = None
	total_start_time = time.time()
	if args.test_only:
		eval_model = get_regression_eval_model(model)
		eval_model.eval()
		with torch.no_grad():
			test_res = evaluation(eval_model, data_obj['test_dataloader'], data_obj['n_test_batches'])
		best_iter = loaded_checkpoint.get('epoch', 0)
		print(f"Test MSE: {test_res['mse']:.5f} Test MAE: {test_res['mae']:.5f}")

	for itr in range(0 if args.test_only else args.epoch):
		st = time.time()
		epoch_train_losses = []
		iter_count = 0
		time_now = time.time()

		### Training ###
		model.train()
		for batch_idx in range(num_batches):
			optimizer.zero_grad()
			# utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			train_res = compute_all_losses(model, batch_dict, args.dataset)
			if not torch.isfinite(train_res['loss']):
				raise FloatingPointError('Regression loss is not finite.')
			train_res["loss"].backward()
			optimizer.step()

			epoch_train_losses.append(train_res["loss"].item())
			iter_count += 1

			if (batch_idx + 1) % 10 == 0:
				speed = (time.time() - time_now) / iter_count
				left_time = speed * (num_batches - (batch_idx + 1))
				print(f"\titers: {batch_idx + 1}, epoch: {itr + 1} | loss: {train_res['loss'].item():.7f}")
				print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
				iter_count = 0
				time_now = time.time()

		epoch_time = time.time() - st
		train_loss_avg = np.mean(epoch_train_losses)
		print(f"Epoch: {itr + 1} cost time: {epoch_time:.2f}s")

		### Validation ###
		model.eval()
		eval_model = get_regression_eval_model(model)
		eval_model.eval()
		with torch.no_grad():
			val_res = evaluation(eval_model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			if not np.isfinite(val_res['mse']):
				raise FloatingPointError('Validation MSE is not finite.')

			### Testing ###
			if(val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				save_regression_checkpoint(checkpoint_path, model, itr)
				print(f'[Info] Saved best checkpoint: {checkpoint_path}')
				test_res = evaluation(eval_model, data_obj["test_dataloader"], data_obj["n_test_batches"])

			vali_loss = val_res["loss"]
			test_loss = test_res["loss"] if test_res is not None else float('inf')
			test_mse  = test_res["mse"]  if test_res is not None else float('inf')
			test_mae  = test_res["mae"]  if test_res is not None else float('inf')

			print(f"Epoch: {itr + 1}, Steps: {num_batches} "
                  f"| Train Loss: {train_loss_avg:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Vali MSE: {val_res['mse']:.5f} "
                  f"Test Loss: {test_loss:.7f} Test MSE: {test_mse:.5f} Test MAE: {test_mae:.5f}")

			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
                     test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s\n".format(time.time()-st))

		if(itr - best_iter >= args.patience):
			print(f"Early stopping triggered. No improvement for {itr - best_iter} epoch(s).")
			training_duration = time.time() - total_start_time
			save_regression_results(args, best_iter, test_res, training_duration)
			sys.exit(0)

		scheduler.step()

	training_duration = time.time() - total_start_time
	save_regression_results(args, best_iter, test_res, training_duration)
