"""Small CPU regressions for public classification and forecasting entry points."""

import os
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

import classification
from lib.Dataset_MM import (
    get_data_mean_std, get_processed_data, get_processed_data_static,
    normalize, variable_time_collate_fn_indseq, variable_time_collate_fn_vector,
)
from lib.evaluation import compute_error
from lib.parse_datasets import task_mask
from lib.ushcn import USHCN_task_mask
from lib.utils import EarlyStopping, evaluate_mc
from models.ITSPM import ITSPM
from regression import BatchOnlyDataParallel, load_regression_checkpoint, save_regression_checkpoint


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def model(self, channels=3):
        return ITSPM(SimpleNamespace(input_dim=channels, d_model=8, dropout=0.,
                                       n_ref_points=2, n_mixer_layers=1, n_scales=2))

    def records(self):
        times = torch.arange(4).float()
        values = torch.arange(12).reshape(4, 3).float()
        return [('patient', times, values, torch.ones_like(values), torch.tensor([1.]))]

    def run_cli(self, command, env):
        output = io.StringIO()
        with patch.object(sys, 'argv', command), patch.dict(os.environ, env), redirect_stdout(output):
            runpy.run_path(command[0], run_name='__main__')
        return output.getvalue()

    def test_model_forward_and_backward(self):
        model = self.model().eval()
        values = torch.randn(2, 4, 3)
        mask = torch.ones_like(values)
        mask[1] = 0
        times = torch.arange(4).float().expand(2, -1)
        self.assertEqual(model(times, values, mask).shape, (2, 8))
        query = torch.tensor([[1.5, 4.], [1.5, 4.]])
        prediction = model.forecasting(query, values, times, mask)
        singleton = model.forecasting(query.unsqueeze(-1), values, times, mask)
        self.assertEqual(prediction.shape, (1, 2, 2, 3))
        torch.testing.assert_close(prediction, singleton)
        prediction.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_invalid_input_shapes_are_rejected(self):
        model = self.model()
        values = torch.ones(2, 4, 3)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 4, 2), values, values)
        with self.assertRaises(ValueError):
            model.forecasting(torch.ones(1, 2), values, torch.ones(2, 4), values)

    def test_tensor_normalization(self):
        values = torch.arange(12).float().reshape(1, 4, 3)
        output = normalize(values, torch.zeros(3, 1), torch.ones(3, 1))
        torch.testing.assert_close(output, values)

    def test_single_observation_statistics(self):
        record = [('id', torch.tensor([0.]), torch.tensor([[2., 0.]]),
                   torch.tensor([[1., 0.]]), torch.tensor([0.]))]
        mean, std, _ = get_data_mean_std(record, torch.device('cpu'))
        self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(std).all())
        self.assertTrue((std > 0).all())

    def test_patient_without_padding(self):
        patient = {'id': 'full', 'time': np.arange(4).reshape(-1, 1),
                   'arr': np.ones((4, 3)), 'static': np.array([1., 2.]),
                   'extended_static': np.array([1., 2.])}
        record = get_processed_data([patient], [[1]])[0]
        self.assertEqual(record[1].shape, (4,))
        static, _ = get_processed_data_static([patient], [[1]], use_static=True)
        self.assertEqual(static[0][2].shape, (4, 5))

    def test_both_classification_collates(self):
        args = SimpleNamespace(fillmiss=False)
        kwargs = dict(args=args, device=torch.device('cpu'), input_dim=3,
                      data_mean=torch.zeros(3), data_std=torch.ones(3), time_max=3.)
        vector, _, lengths = variable_time_collate_fn_vector(self.records(), maxlen=2, **kwargs)
        self.assertEqual(vector.shape, (1, 2, 9))
        self.assertEqual(lengths.numel(), 1)
        individual, _, lengths = variable_time_collate_fn_indseq(self.records(), maxlen=2, **kwargs)
        self.assertEqual(individual.shape, (1, 2, 9))
        self.assertEqual(lengths.item(), 2)

    def test_classifier_evaluation_disables_dropout(self):
        class Backbone(nn.Module):
            def forward(self, times, values, mask, opt):
                return values.mean(dim=1)

        values = torch.randn(4, 3, 2)
        batch = torch.cat([values, torch.ones_like(values), torch.zeros_like(values)], dim=-1)
        labels = torch.tensor([0, 1, 0, 1])
        data = [(batch, labels, torch.full((4,), 3))]
        args = SimpleNamespace(device=torch.device('cpu'), num_types=2, n_classes=2, focal_gamma=0.)
        classifier = classification.ITSPMClassifier(2, 2, dropout=.9).train()
        loss = nn.CrossEntropyLoss(reduction='none')
        first = classification.eval_epoch(Backbone(), data, loss, args, classifier)
        second = classification.eval_epoch(Backbone(), data, loss, args, classifier)
        self.assertFalse(classifier.training)
        np.testing.assert_allclose(first, second)

    def test_retrain_unpacking(self):
        args = SimpleNamespace(model='ITSPM', test_only=False, retrain=True, epoch=1)
        with patch.object(classification, 'train_epoch', return_value=(.5, .6, .7, 1., 1)), \
             patch.object(classification, 'eval_epoch', return_value=(.5, .6, .7, .8, .9, .6, 1.)), \
             patch.object(classification, 'log_info'):
            classification.run_experiment(nn.Linear(2, 2), [], [], [], Mock(), Mock(),
                                          Mock(), args, classifier=nn.Linear(2, 2))

    def test_partial_class_metrics(self):
        for classes, labels in [(2, np.array([0, 0])), (8, np.array([2, 2])), (8, np.array([1, 2]))]:
            result = evaluate_mc(labels, np.zeros((2, classes)), classes)
            self.assertTrue(np.isfinite(result[0]))
            self.assertTrue(np.isfinite(result[2]))
            self.assertTrue(np.isnan(result[1]))

    def test_uneven_multi_gpu_scatter(self):
        inputs = (torch.ones(5, 2), torch.ones(5, 4, 3),
                  torch.ones(5, 4), torch.ones(5, 4, 3))
        with patch.object(torch.Tensor, 'to', lambda tensor, *args, **kwargs: tensor):
            chunks, _ = BatchOnlyDataParallel(nn.Identity()).scatter(inputs, {}, [0, 1, 2, 3])
        self.assertEqual(len(chunks), 4)
        self.assertEqual([chunk[1].size(0) for chunk in chunks], [2, 1, 1, 1])

    def test_error_metrics(self):
        truth = torch.tensor([[[-2.]]])
        prediction = torch.tensor([[[0.]]], requires_grad=True)
        mask = torch.ones_like(truth)
        self.assertEqual(compute_error(truth, prediction, mask, 'MAPE', 'mean').item(), 1.)
        compute_error(truth, prediction, mask, 'HUBER', 'mean').backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        with self.assertRaises(ValueError):
            compute_error(truth, prediction, torch.zeros_like(mask), 'MSE', 'mean')

    def test_empty_windows_and_masks(self):
        args = SimpleNamespace(task='forecasting', history=2, dataset='activity')
        valid = self.records()[0][:4]
        invalid = ('empty', torch.arange(4).float(), torch.ones(4, 3), torch.zeros(4, 3))
        self.assertEqual(len(task_mask(args, [valid, invalid])), 1)
        self.assertEqual(len(task_mask(args, [valid, invalid], filter_invalid=False)), 2)
        with self.assertRaises(ValueError):
            task_mask(args, [invalid])
        with self.assertRaises(ValueError):
            USHCN_task_mask(args, [invalid + (torch.tensor(0.),)])

    def test_checkpoint_roundtrip(self):
        model = self.model()
        stopping = EarlyStopping()
        self.assertTrue(np.isinf(stopping.val_loss_min))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            save_regression_checkpoint(path, model, 3)
            replacement = self.model()
            checkpoint = load_regression_checkpoint(path, replacement)
            self.assertEqual(checkpoint['epoch'], 3)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, replacement.state_dict()[key])

    def test_synthetic_regression_cli_training_and_test_only(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / 'data' / 'mimic'
            data.mkdir(parents=True)
            records = []
            for i in range(16):
                times = torch.arange(4).float()
                values = torch.tensor([[i + t, i - t] for t in range(4)]).float()
                records.append((str(i), times, values, torch.ones_like(values)))
            torch.save(records, data / 'mimic.pt')
            command = [str(root / 'regression.py'), '--dataset', 'mimic',
                       '--task', 'forecasting', '--data_path', str(tmp / 'data'), '--epoch', '1',
                       '--batch_size', '4', '--history', '2', '--d_model', '8', '--n_mixer_layers', '0',
                       '--save_path', str(tmp / 'checkpoints'), '--log', str(tmp / 'logs')]
            env = dict(os.environ, ITSPM_REGRESSION_RESULTS_FILE=str(tmp / 'results.txt'))
            self.run_cli(command, env)
            checkpoint = next((tmp / 'checkpoints').glob('*.pt'))
            evaluated = self.run_cli(command + ['--test_only', '--load_path', str(checkpoint)], env)
            self.assertNotIn('Epoch:', evaluated)
            self.assertIn('Test MSE:', evaluated)

    def test_synthetic_classification_cli_training_and_test_only(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / 'data' / 'P12data'
            (data / 'processed_data').mkdir(parents=True)
            (data / 'splits').mkdir()
            records = [{'id': str(i), 'time': np.arange(4).reshape(-1, 1),
                        'arr': np.ones((4, 3)) * (i + 1)} for i in range(12)]
            np.save(data / 'processed_data' / 'PTdict_list.npy', np.array(records, dtype=object))
            np.save(data / 'processed_data' / 'arr_outcomes.npy', np.arange(12).reshape(-1, 1) % 2)
            splits = np.empty(3, dtype=object)
            splits[:] = [np.arange(8), np.arange(8, 10), np.arange(10, 12)]
            np.save(data / 'splits' / 'phy12_split1.npy', splits)
            command = [str(root / 'classification.py'), '--task', 'P12', '--data_path', str(tmp / 'data'),
                       '--epoch', '1', '--batch_size', '4', '--d_model', '8', '--n_mixer_layers', '0',
                       '--save_path', str(tmp / 'checkpoints') + '/', '--log', str(tmp / 'logs') + '/']
            env = dict(ITSPM_CLASSIFICATION_RESULTS_FILE=str(tmp / 'results.txt'))
            self.run_cli(command, env)
            checkpoint = next((tmp / 'checkpoints').glob('*.h5'))
            modified = checkpoint.stat().st_mtime_ns
            evaluated = self.run_cli(command + ['--test_only', '--load_path', str(checkpoint)], env)
            self.assertNotIn('[ Epoch', evaluated)
            self.assertEqual(checkpoint.stat().st_mtime_ns, modified)


if __name__ == '__main__':
    unittest.main()
