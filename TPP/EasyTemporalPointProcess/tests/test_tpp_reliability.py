"""Small CPU regression checks for TPP masks, likelihoods, and data boundaries."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from easy_tpp.config_factory import DataConfig, DataSpecConfig, ModelConfig
from easy_tpp.config_factory.model_config import TrainerConfig
from easy_tpp.model.torch_model.torch_itspm import ITSPM, _PatternInteraction
from easy_tpp.preprocess.data_loader import TPPDataLoader
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.torch_wrapper import TorchModelWrapper
from easy_tpp.utils.torch_utils import set_device


def model_config(event_types=2, thinning=None):
    return ModelConfig(
        model_id='ITSPM', hidden_size=8, num_event_types=event_types,
        num_event_types_pad=event_types + 1, event_pad_index=event_types,
        gpu=-1, dropout_rate=0., loss_integral_num_sample_per_step=3,
        num_layers=1, num_heads=2, thinning=thinning,
        model_specs={'window_size': 4, 'max_event_tokens': 4, 'max_gap_tokens': 2},
    )


def tokenizer(side='right', event_types=2):
    return EventTokenizer(DataSpecConfig(
        num_event_types=event_types, pad_token_id=event_types, padding_side=side,
        truncation_side='right',
    ))


class TPPReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_log_intensity_still_counts_event_and_respects_mask(self):
        model = ITSPM(model_config())
        intensity = torch.full((1, 3, 2), 1. - model.eps)
        event, integral, count = model.compute_loglikelihood(
            torch.ones(1, 3), intensity, intensity.unsqueeze(2).expand(-1, -1, 3, -1),
            torch.tensor([[True, False, True]]), torch.tensor([[0, 1, 2]]),
        )
        self.assertEqual(count, 1)
        torch.testing.assert_close(event, torch.zeros_like(event))
        torch.testing.assert_close(integral, torch.tensor([[2., 0., 0.]]))

    def test_interval_stats_match_direct_population_statistics(self):
        data = TPPDataset({
            'time_seqs': [[0., 1., 3.], [0., 9.]],
            'time_delta_seqs': [[0., 1., 2.], [0., 9.]],
            'type_seqs': [[0, 1, 0], [1, 0]],
        })
        mean, std, minimum, maximum = data.get_dt_stats()
        np.testing.assert_allclose([mean, std, minimum, maximum],
                                   [4., np.std([1., 2., 9.]), 1., 9.])
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            TPPDataset({'time_seqs': [[0., 1.]], 'time_delta_seqs': [[0.]], 'type_seqs': [[0, 1]]})
        with self.assertRaisesRegex(ValueError, 'interval'):
            TPPDataset({'time_seqs': [[0.]], 'time_delta_seqs': [[0.]], 'type_seqs': [[0]]}).get_dt_stats()

    def test_left_and_right_padding_have_identical_valid_likelihoods(self):
        data = {'time_seqs': [[0., 1., 3.], [0., 2.]],
                'time_delta_seqs': [[0., 1., 2.], [0., 2.]],
                'type_seqs': [[0, 1, 0], [1, 0]]}
        left = tokenizer('left').pad(copy.deepcopy(data), return_tensors='pt')
        right = tokenizer().pad(copy.deepcopy(data), return_tensors='pt')
        self.assertEqual(left['seq_non_pad_mask'].tolist(), [[True, True, True], [False, True, True]])
        model = ITSPM(model_config()).eval()
        left_loss, left_count = model.loglike_loss(left.values())
        right_loss, right_count = model.loglike_loss(right.values())
        self.assertEqual((left_count, right_count), (3, 3))
        torch.testing.assert_close(left_loss, right_loss)
        right_loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_future_events_do_not_change_prefix_states(self):
        model = ITSPM(model_config()).eval()
        times = torch.tensor([[0., 1., 3., 5.]])
        marks = torch.tensor([[0, 1, 0, 1]])
        original = model(times, marks)
        times[:, -1] = 100.
        marks[:, -1] = 0
        changed = model(times, marks)
        torch.testing.assert_close(original[:, :-1], changed[:, :-1])

    def test_event_and_gap_pooling_use_separate_blocks_and_masks(self):
        interaction = _PatternInteraction(
            num_channels=2, d_model=4, dropout=0., n_layers=0, n_heads=1,
            max_event_tokens=2, max_gap_tokens=2,
        )
        # Masked outliers must not leak into any type-specific summary.
        tokens = torch.tensor([[[1.] * 4, [999.] * 4,
                                [10.] * 4, [888.] * 4,
                                [100.] * 4, [777.] * 4]])
        mask = torch.tensor([[True, False, True, False, True, False]])
        captured = {}
        handles = []
        for index, branch in enumerate(interaction.type_proj):
            def capture(module, args, branch_index=index):
                captured[branch_index] = args[0].detach().clone()
            handles.append(branch.register_forward_pre_hook(capture))
        try:
            interaction(tokens, mask)
        finally:
            for handle in handles:
                handle.remove()
        for index, expected in enumerate((1., 10., 100.)):
            torch.testing.assert_close(captured[index], torch.full((1, 8), expected))
        self.assertFalse(torch.equal(captured[0], captured[1]))

    def test_last_step_intensities_accept_last_step_sample_shape(self):
        model = ITSPM(model_config(thinning={
            'num_sample': 1, 'num_exp': 4, 'num_samples_boundary': 2, 'dtime_max': 1.,
        })).eval()
        times = torch.tensor([[0., 1., 3.]])
        marks = torch.tensor([[0, 1, 0]])
        deltas = torch.tensor([[0., 1., 2.]])
        samples = torch.tensor([[[0., 0.5, 1.]]])
        result = model.compute_intensities_at_sample_times(
            times, deltas, marks, samples, compute_last_step_only=True,
        )
        self.assertEqual(result.shape, (1, 1, 3, 2))
        self.assertTrue(torch.isfinite(result).all())
        batch = tokenizer().pad({'time_seqs': times.tolist(), 'time_delta_seqs': deltas.tolist(),
                                 'type_seqs': marks.tolist()}, return_tensors='pt')
        prediction = model.predict_multi_step_since_last_event(batch.values())
        self.assertTrue(all(torch.isfinite(value).all() for value in prediction))
        self.assertEqual(prediction[0].shape, prediction[2].shape)

    def test_config_copy_preserves_padding_device_and_thinning(self):
        specs = DataSpecConfig(num_event_types=2, pad_token_id=2, padding_side='right')
        cfg = DataConfig('train.pkl', 'dev.pkl', 'test.pkl', 'pkl', specs)
        copied = cfg.copy()
        self.assertEqual(copied.data_format, 'pkl')
        self.assertEqual(copied.data_specs.pad_token_id, 2)
        self.assertIsNot(copied.data_specs, specs)
        trainer = TrainerConfig(seed=123, gpu=-1)
        self.assertEqual((trainer.copy().seed, trainer.copy().gpu), (123, -1))
        model = model_config(thinning={'num_exp': 4})
        self.assertEqual(model.copy().thinning.num_exp, 4)
        self.assertEqual(model.copy().model_id, 'ITSPM')

    def test_checkpoint_restore_is_strict_and_available_on_cpu(self):
        config = model_config()
        config.is_training = False
        original = ITSPM(config)
        wrapper = TorchModelWrapper(original, SimpleNamespace(model_id='ITSPM'), config,
                                    TrainerConfig(gpu=-1, use_tfb=False))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.pt'
            wrapper.save(checkpoint)
            with torch.no_grad():
                original.factor_intensity_base.add_(10.)
            wrapper.restore(checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            torch.testing.assert_close(original.factor_intensity_base, state['factor_intensity_base'])
            state.pop('factor_intensity_base')
            torch.save(state, checkpoint)
            with self.assertRaises(RuntimeError):
                wrapper.restore(checkpoint)

    def test_unavailable_gpu_produces_clear_error(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('easy_tpp.utils.torch_utils.is_torch_mps_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU'):
                set_device(0)

    def test_taxi_real_batch_has_finite_loss_without_training(self):
        specs = DataSpecConfig(num_event_types=10, pad_token_id=10, padding_side='right')
        data_root = ROOT.parent / 'taxi'
        cfg = DataConfig(str(data_root / 'train.pkl'), str(data_root / 'dev.pkl'),
                         str(data_root / 'test.pkl'), 'pkl', specs)
        loader = TPPDataLoader(cfg, backend='torch', batch_size=2, shuffle=False).train_loader()
        batch = next(iter(loader))
        model = ITSPM(model_config(event_types=10)).eval()
        loss, count = model.loglike_loss(batch.values())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(count, int(batch['seq_non_pad_mask'][:, 1:].sum()))


if __name__ == '__main__':
    unittest.main()
