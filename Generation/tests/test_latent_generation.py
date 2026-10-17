"""Behavioral checks for the generation head, without training experiments."""
import argparse
import contextlib
import inspect
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Generation.itspm_generation import train_irregular_generation as generation


class LatentGenerationTests(unittest.TestCase):
    def make_models(self):
        cfg = generation.ITSPMConfig(
            input_dim=2, d_model=8, dropout=0.0, n_ref_points=4,
            n_scales=1, n_mixer_layers=1, max_event_tokens=4, max_gap_tokens=2,
        )
        args = argparse.Namespace(
            seq_len=4, channels=2, latent_dim=4, diffusion_hidden=8,
            diffusion_steps=2, beta_start=1e-4, beta_end=0.02,
        )
        ae = generation.ITSPMLatentAutoencoder(cfg, 4, 2, 4)
        denoiser = generation.LatentDenoiser(4, 8, 2)
        ddpm = generation.LatentDDPM(2, 1e-4, 0.02, torch.device("cpu"))
        return cfg, args, ae, denoiser, ddpm

    def test_missing_targets_do_not_affect_loss_or_gradients(self):
        reconstruction = torch.tensor([[[3.0, 7.0], [10.0, 5.0]]], requires_grad=True)
        mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[1.0, float("nan")], [float("inf"), 2.0]]])
        loss = generation.observed_reconstruction_loss(reconstruction, target, mask)
        self.assertAlmostEqual(loss.item(), 6.5)
        loss.backward()
        torch.testing.assert_close(reconstruction.grad, torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]))
        target[mask == 0] = -99999.0
        self.assertAlmostEqual(
            generation.observed_reconstruction_loss(reconstruction, target, mask).item(), 6.5,
        )
        with self.assertRaises(ValueError):
            generation.observed_reconstruction_loss(reconstruction, target, torch.zeros_like(mask))

    def test_standardizer_ignores_hidden_training_and_all_evaluation_values(self):
        data = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        mask = np.ones_like(data)
        mask[:2, 1, :] = 0.0
        normalized, mean, std, ids = generation.normalize_train_test(data, 0.5, mask)
        altered = data.copy()
        altered[:2, 1, :] = np.nan
        altered[2:] = 1e9
        changed, changed_mean, changed_std, changed_ids = generation.normalize_train_test(altered, 0.5, mask)
        np.testing.assert_array_equal(ids, changed_ids)
        np.testing.assert_array_equal(mean, changed_mean)
        np.testing.assert_array_equal(std, changed_std)
        visible = mask[:2] > 0
        np.testing.assert_array_equal(normalized[:2][visible], changed[:2][visible])
        np.testing.assert_allclose(generation.denormalize(normalized, mean, std), data, atol=1e-6)
        for fraction in (0.0, 1.0):
            with self.assertRaises(ValueError):
                generation.normalize_train_test(data, fraction, mask)

    def test_latent_standardization_round_trip_and_singleton(self):
        for latents in (torch.tensor([[2.0, 4.0], [6.0, 4.0]]), torch.tensor([[2.0, 4.0]])):
            normalized, mean, std = generation.standardize_training_latents(latents)
            self.assertTrue(torch.isfinite(normalized).all())
            torch.testing.assert_close(normalized * std + mean, latents)

    def test_freezing_disables_dropout_gradients_and_encoder_sampling(self):
        _, _, ae, denoiser, ddpm = self.make_models()
        ae.train()
        for parameter in ae.parameters():
            parameter.grad = torch.ones_like(parameter)
        generation.freeze_autoencoder(ae)
        self.assertFalse(ae.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ae.parameters()))
        with patch.object(ae, "encode", side_effect=AssertionError("Sampling must not encode observations")):
            output = generation.sample_sequences(ae, denoiser, ddpm, 3, torch.zeros(1, 4), torch.ones(1, 4))
        self.assertEqual(tuple(output.shape), (3, 4, 2))
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.isfinite(output).all())
        self.assertNotIn("mask", inspect.signature(generation.sample_sequences).parameters)
        self.assertNotIn("obs", inspect.signature(generation.sample_sequences).parameters)

    def test_inverse_latent_transform_precedes_decoding(self):
        _, _, ae, denoiser, ddpm = self.make_models()
        noise_sample = torch.full((2, 4), 2.0)
        mean, std = torch.full((1, 4), 3.0), torch.full((1, 4), 5.0)
        with patch.object(ddpm, "sample", return_value=noise_sample), patch.object(ae, "decode", wraps=ae.decode) as decode:
            generation.sample_sequences(ae, denoiser, ddpm, 2, mean, std)
        torch.testing.assert_close(decode.call_args.args[0], torch.full((2, 4), 13.0))

    def test_checkpoint_sampling_needs_no_dataset_and_preserves_statistics(self):
        cfg, args, ae, denoiser, ddpm = self.make_models()
        mean = np.array([[[2.0, -1.0]]], dtype=np.float32)
        std = np.array([[[3.0, 4.0]]], dtype=np.float32)
        latent_mean = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        latent_std = torch.tensor([[0.5, 0.8, 1.2, 1.5]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            generation.save_generation_checkpoint(path, ae, denoiser, cfg, args, mean, std, latent_mean, latent_std)
            generation.set_seed(9)
            normalized = generation.sample_sequences(ae, denoiser, ddpm, 3, latent_mean, latent_std).numpy()
            expected = generation.denormalize(normalized, mean, std)
            with patch.object(generation, "load_table1_dataset", side_effect=AssertionError("No dataset needed")):
                actual = generation.sample_from_checkpoint(path, 3, torch.device("cpu"), seed=9)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_cli_defaults_to_latent_and_rejects_retired_modes(self):
        with patch.object(sys, "argv", ["generation"]):
            args = generation.parse_args()
        self.assertEqual(args.generator, "itspm_latent_diffusion")
        for option in ("--full_recon_weight", "--calibrate_marginals", "--clamp_observed"):
            with patch.object(sys, "argv", ["generation", option]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    generation.parse_args()
        with patch.object(sys, "argv", ["generation", "--generator", "itspm_conditioned_diffusion"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                generation.parse_args()

    def test_mmd_matches_pairwise_reference_and_handles_constant_samples(self):
        rng = np.random.default_rng(5)
        real, fake = rng.normal(size=(5, 4, 2)), rng.normal(size=(7, 4, 2))
        x, y = real.reshape(5, -1), fake.reshape(7, -1)
        xy = np.vstack([x, y])
        sq = np.square(xy[:, None] - xy[None, :]).sum(axis=-1)
        gamma = 1.0 / (2.0 * (np.median(sq[sq > 0]) + 1e-6))
        kernel = np.exp(-gamma * sq)
        expected = kernel[:5, :5].mean() + kernel[5:, 5:].mean() - 2 * kernel[:5, 5:].mean()
        self.assertAlmostEqual(generation.rbf_mmd(real, fake), expected, places=12)
        constant = np.ones((3, 4, 2), dtype=np.float32)
        self.assertEqual(generation.rbf_mmd(constant, constant), 0.0)

    def test_short_sequence_and_out_of_range_metrics_remain_finite(self):
        real = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        fake = real + 1000.0
        expected_acf = np.mean([
            np.abs((real[:, :-lag] * real[:, lag:]).mean(axis=(0, 1))
                   - (fake[:, :-lag] * fake[:, lag:]).mean(axis=(0, 1))).mean()
            for lag in range(1, 4)
        ])
        self.assertAlmostEqual(generation.acf_error(real, fake), float(expected_acf))
        self.assertTrue(np.isfinite(generation.flat_kl(real, fake)))

    def test_metrics_reject_empty_train_or_time_pairs(self):
        singleton = np.ones((1, 4, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "at least two real"):
            generation.table1_discriminative_score(singleton, singleton, torch.device("cpu"), 1, 1)
        with self.assertRaisesRegex(ValueError, "at least two timesteps"):
            generation.acf_error(singleton[:, :1], singleton[:, :1])


if __name__ == "__main__":
    unittest.main()
