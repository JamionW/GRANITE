"""
Regression tests for loss term machinery after smoothness term removal and
accessibility_consistency_loss rename (audit followup 2026-05-27).

Verifies:
- loss dict contains expected keys
- loss dict does not contain 'smoothness' or old 'accessibility' key
- loading a config with smoothness_weight raises ValueError (fail-fast guard)
- loading a config with variation_weight wires self.variation_weight in both trainers
- MultiTractGNNTrainer defaults variation_weight to 0.8; AccessibilityGNNTrainer defaults to 1.5
- _compute_min_spread_loss (formerly _compute_accessibility_consistency_loss) is callable
"""
import pytest
import torch

from granite.models.gnn import AccessibilitySVIGNN, MultiTractGNNTrainer, AccessibilityGNNTrainer


def _make_multi_trainer(config=None):
    model = AccessibilitySVIGNN(
        accessibility_features_dim=10,
        context_features_dim=5,
        hidden_dim=16,
        seed=0,
    )
    return MultiTractGNNTrainer(model, config=config or {}, seed=0)


def _make_single_trainer(config=None):
    model = AccessibilitySVIGNN(
        accessibility_features_dim=10,
        context_features_dim=5,
        hidden_dim=16,
        seed=0,
    )
    return AccessibilityGNNTrainer(model, config=config or {}, seed=0)


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
    EXPECTED_MULTI_KEYS = {'total', 'constraint', 'variation', 'bounds', 'bg_constraint', 'ordering'}
    EXPECTED_SINGLE_KEYS = {'total', 'constraint', 'variation', 'bounds', 'range', 'min_spread'}

    def test_multi_tract_expected_keys_present(self):
        trainer = _make_multi_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert set(losses.keys()) == self.EXPECTED_MULTI_KEYS, (
            f"loss dict keys changed: got {set(losses.keys())}, expected {self.EXPECTED_MULTI_KEYS}"
        )

    def test_multi_tract_smoothness_key_absent(self):
        trainer = _make_multi_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert 'smoothness' not in losses, (
            "'smoothness' key found in loss dict; the term should have been removed"
        )

    def test_multi_tract_old_accessibility_key_absent(self):
        # key was 'accessibility' before rename; must now be 'min_spread'
        trainer = _make_multi_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert 'accessibility' not in losses, (
            "'accessibility' key still present; should have been renamed to 'min_spread'"
        )

    def test_multi_tract_total_loss_is_finite(self):
        trainer = _make_multi_trainer()
        predictions, tract_targets, tract_masks = _make_inputs()
        losses = trainer._compute_multi_tract_losses(predictions, tract_targets, tract_masks, n_addresses=12)
        assert torch.isfinite(losses['total']), "total loss is not finite"

    def test_single_tract_expected_keys_present(self):
        trainer = _make_single_trainer()
        predictions = torch.rand(12).clamp(0.01, 0.99)
        target_svi = torch.tensor([0.4])
        losses = trainer._compute_losses(predictions, target_svi, n_addresses=12)
        assert set(losses.keys()) == self.EXPECTED_SINGLE_KEYS, (
            f"single-tract loss dict keys changed: got {set(losses.keys())}, "
            f"expected {self.EXPECTED_SINGLE_KEYS}"
        )

    def test_single_tract_min_spread_key_present(self):
        trainer = _make_single_trainer()
        predictions = torch.rand(12).clamp(0.01, 0.99)
        target_svi = torch.tensor([0.4])
        losses = trainer._compute_losses(predictions, target_svi, n_addresses=12)
        assert 'min_spread' in losses, "'min_spread' key missing from single-tract loss dict"

    def test_single_tract_old_accessibility_key_absent(self):
        trainer = _make_single_trainer()
        predictions = torch.rand(12).clamp(0.01, 0.99)
        target_svi = torch.tensor([0.4])
        losses = trainer._compute_losses(predictions, target_svi, n_addresses=12)
        assert 'accessibility' not in losses, (
            "'accessibility' key still present in single-tract losses; "
            "should have been renamed to 'min_spread'"
        )


class TestSmoothnessWeightFailFast:
    def test_smoothness_weight_in_config_raises(self):
        with pytest.raises(ValueError, match="smoothness_weight"):
            _make_multi_trainer(config={'smoothness_weight': 0.1})

    def test_smoothness_weight_zero_also_raises(self):
        # even weight=0.0 should raise; the key itself is disallowed
        with pytest.raises(ValueError, match="smoothness_weight"):
            _make_multi_trainer(config={'smoothness_weight': 0.0})

    def test_clean_config_does_not_raise(self):
        # a config without smoothness_weight must succeed
        trainer = _make_multi_trainer(config={'constraint_weight': 2.0})
        assert trainer is not None


class TestVariationWeightWiring:
    """variation_weight is now a valid MultiTractGNNTrainer config key (step 4b)."""

    def test_multi_trainer_reads_variation_weight(self):
        trainer = _make_multi_trainer(config={'variation_weight': 1.5})
        assert trainer.variation_weight == pytest.approx(1.5), (
            "variation_weight not wired: expected self.variation_weight=1.5"
        )

    def test_multi_trainer_default_variation_weight(self):
        trainer = _make_multi_trainer(config={})
        assert trainer.variation_weight == pytest.approx(0.8), (
            "default variation_weight should be 0.8 for backward compatibility"
        )

    def test_multi_trainer_explicit_default_matches_implicit(self):
        t_explicit = _make_multi_trainer(config={'variation_weight': 0.8})
        t_implicit = _make_multi_trainer(config={})
        assert t_explicit.variation_weight == pytest.approx(t_implicit.variation_weight)

    def test_multi_trainer_clean_config_does_not_raise(self):
        trainer = _make_multi_trainer(config={'constraint_weight': 2.0})
        assert trainer is not None

    def test_single_trainer_reads_variation_weight(self):
        trainer = _make_single_trainer(config={'variation_weight': 2.5})
        assert trainer.variation_weight == pytest.approx(2.5), (
            "variation_weight not wired in AccessibilityGNNTrainer: expected 2.5"
        )

    def test_single_trainer_default_variation_weight(self):
        trainer = _make_single_trainer(config={})
        assert trainer.variation_weight == pytest.approx(1.5), (
            "default variation_weight for single-tract trainer should be 1.5"
        )


class TestMinSpreadLoss:
    def test_min_spread_callable(self):
        trainer = _make_single_trainer()
        predictions = torch.rand(12).clamp(0.01, 0.99)
        loss = trainer._compute_min_spread_loss(predictions)
        assert torch.isfinite(loss), "_compute_min_spread_loss returned non-finite value"

    def test_min_spread_zero_for_diverse_predictions(self):
        trainer = _make_single_trainer()
        # predictions uniformly spread: mean consecutive diff >> 0.001, hinge inactive
        predictions = torch.linspace(0.1, 0.9, 20)
        loss = trainer._compute_min_spread_loss(predictions)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            "min_spread_loss should be zero when predictions are already well-spread"
        )

    def test_min_spread_positive_for_collapsed_predictions(self):
        trainer = _make_single_trainer()
        # all predictions identical: sorted gradient is zero, hinge fires
        predictions = torch.full((12,), 0.5)
        loss = trainer._compute_min_spread_loss(predictions)
        assert loss.item() > 0.0, (
            "min_spread_loss should be positive when predictions are collapsed"
        )

    def test_min_spread_returns_zero_for_small_inputs(self):
        trainer = _make_single_trainer()
        predictions = torch.rand(3)
        loss = trainer._compute_min_spread_loss(predictions)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            "min_spread_loss should return 0 when len(predictions) < 4"
        )
