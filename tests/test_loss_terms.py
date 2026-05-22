"""
Regression tests for _compute_multi_tract_losses after smoothness term removal.

Verifies:
- loss dict contains expected keys
- loss dict does not contain 'smoothness'
- loading a config with smoothness_weight raises ValueError (fail-fast guard)
"""
import pytest
import torch

from granite.models.gnn import AccessibilitySVIGNN, MultiTractGNNTrainer


def _make_trainer(config=None):
    model = AccessibilitySVIGNN(
        accessibility_features_dim=10,
        context_features_dim=5,
        hidden_dim=16,
        seed=0,
    )
    return MultiTractGNNTrainer(model, config=config or {}, seed=0)


def _make_inputs(n=12, n_tracts=2):
    """minimal predictions + tract_masks for loss computation."""
    predictions = torch.rand(n).clamp(0.01, 0.99)
    tract_masks = {}
    chunk = n // n_tracts
    for t in range(n_tracts):
        start = t * chunk
        end = start + chunk if t < n_tracts - 1 else n
        mask = torch.zeros(n, dtype=torch.bool)
        mask[start:end] = True
        tract_masks[f'tract_{t}'] = mask
    tract_targets = {k: torch.tensor([0.3 + t * 0.2]) for t, k in enumerate(tract_masks)}
    return predictions, tract_targets, tract_masks


class TestLossDictKeys:
    EXPECTED_KEYS = {'total', 'constraint', 'variation', 'bounds', 'bg_constraint', 'ordering'}

    def test_expected_keys_present(self):
        trainer = _make_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert set(losses.keys()) == self.EXPECTED_KEYS, (
            f"loss dict keys changed: got {set(losses.keys())}, expected {self.EXPECTED_KEYS}"
        )

    def test_smoothness_key_absent(self):
        trainer = _make_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert 'smoothness' not in losses, (
            "'smoothness' key found in loss dict; the term should have been removed"
        )

    def test_total_loss_is_finite(self):
        trainer = _make_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert torch.isfinite(losses['total']), "total loss is not finite"


class TestSmoothessWeightFailFast:
    def test_smoothness_weight_in_config_raises(self):
        with pytest.raises(ValueError, match="smoothness_weight"):
            _make_trainer(config={'smoothness_weight': 0.1})

    def test_smoothness_weight_zero_also_raises(self):
        # even weight=0.0 should raise; the key itself is disallowed
        with pytest.raises(ValueError, match="smoothness_weight"):
            _make_trainer(config={'smoothness_weight': 0.0})

    def test_clean_config_does_not_raise(self):
        # a config without smoothness_weight must succeed
        trainer = _make_trainer(config={'constraint_weight': 2.0})
        assert trainer is not None
