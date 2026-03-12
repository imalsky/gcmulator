"""Regression tests for CPU-side pair resampling from saved sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gcmulator.normalization import ParamNormalizationStats
from gcmulator.sampling import LiveTransitionCatalog
from gcmulator.training import (
    PreloadedSequenceSplit,
    ResampledSplitPlan,
    SequencePairSelection,
    _build_resampled_split_plan,
    _iter_resampled_pair_batches,
    _iter_resampled_pair_batches_preloaded,
    _sample_sequence_pair_selection,
)


def _transition_time_stats() -> ParamNormalizationStats:
    """Return simple transition-time normalization stats for unit tests."""
    return ParamNormalizationStats(
        param_names=("log10_transition_days",),
        mean=np.asarray([0.0], dtype=np.float64),
        std=np.asarray([1.0], dtype=np.float64),
        is_constant=np.asarray([False], dtype=bool),
        zscore_eps=1.0e-8,
    )


def _catalog() -> LiveTransitionCatalog:
    """Return a small variable-gap catalog with nontrivial anchor counts."""
    return LiveTransitionCatalog(
        gap_offsets=np.asarray([1, 2], dtype=np.int64),
        transition_days=np.asarray([0.1, 0.2], dtype=np.float64),
        probabilities=np.asarray([0.5, 0.5], dtype=np.float64),
        burn_in_start_index=1,
    )


def test_sample_sequence_pair_selection_covers_all_valid_candidates() -> None:
    """Sampling the full candidate budget should recover every valid pair exactly once."""
    selection = _sample_sequence_pair_selection(
        sequence_length=6,
        catalog=_catalog(),
        pairs_per_sim=7,
        pair_sampling_policy="uniform_pairs",
        transition_time_stats=_transition_time_stats(),
        seed=11,
    )

    actual_pairs = {
        (int(anchor), int(target))
        for anchor, target in zip(
            selection.anchor_indices.tolist(),
            selection.target_indices.tolist(),
            strict=True,
        )
    }
    expected_pairs = {
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (1, 3),
        (2, 4),
        (3, 5),
    }

    assert actual_pairs == expected_pairs
    assert selection.transition_time_norm.shape == (7, 1)
    for anchor, target, transition_days in zip(
        selection.anchor_indices.tolist(),
        selection.target_indices.tolist(),
        selection.transition_days.tolist(),
        strict=True,
    ):
        realized_gap = int(target) - int(anchor)
        expected_days = 0.1 if realized_gap == 1 else 0.2
        assert transition_days == pytest.approx(expected_days)


def test_sample_sequence_pair_selection_rejects_too_many_pairs() -> None:
    """The resampled mode should fail fast when the requested pair budget is impossible."""
    with pytest.raises(ValueError, match="pairs_per_sim exceeds"):
        _sample_sequence_pair_selection(
            sequence_length=6,
            catalog=_catalog(),
            pairs_per_sim=8,
            pair_sampling_policy="uniform_pairs",
            transition_time_stats=_transition_time_stats(),
            seed=0,
        )


def test_uniform_pairs_sampling_is_uniform_over_candidate_space() -> None:
    """Uniform-pair mode should treat every valid anchor-target pair equally."""
    candidate_counts = {
        (1, 2): 0,
        (2, 3): 0,
        (3, 4): 0,
        (4, 5): 0,
        (1, 3): 0,
        (2, 4): 0,
        (3, 5): 0,
    }

    for seed in range(7000):
        selection = _sample_sequence_pair_selection(
            sequence_length=6,
            catalog=_catalog(),
            pairs_per_sim=1,
            pair_sampling_policy="uniform_pairs",
            transition_time_stats=_transition_time_stats(),
            seed=seed,
        )
        pair = (int(selection.anchor_indices[0]), int(selection.target_indices[0]))
        candidate_counts[pair] += 1

    observed = np.asarray(list(candidate_counts.values()), dtype=np.float64)
    expected = np.mean(observed)
    assert np.all(np.abs(observed - expected) < 0.15 * expected)


def test_build_resampled_split_plan_is_deterministic_for_fixed_seed() -> None:
    """Validation/test resampling plans should be reproducible for a fixed seed."""
    plan_a = _build_resampled_split_plan(
        n_sequences=3,
        sequence_length=6,
        catalog=_catalog(),
        pairs_per_sim=3,
        pair_sampling_policy="uniform_pairs",
        transition_time_stats=_transition_time_stats(),
        seed=17,
        shuffle_sequences=False,
    )
    plan_b = _build_resampled_split_plan(
        n_sequences=3,
        sequence_length=6,
        catalog=_catalog(),
        pairs_per_sim=3,
        pair_sampling_policy="uniform_pairs",
        transition_time_stats=_transition_time_stats(),
        seed=17,
        shuffle_sequences=False,
    )
    plan_c = _build_resampled_split_plan(
        n_sequences=3,
        sequence_length=6,
        catalog=_catalog(),
        pairs_per_sim=3,
        pair_sampling_policy="uniform_pairs",
        transition_time_stats=_transition_time_stats(),
        seed=18,
        shuffle_sequences=False,
    )

    assert np.array_equal(plan_a.sequence_order, np.asarray([0, 1, 2], dtype=np.int64))
    assert np.array_equal(plan_a.sequence_order, plan_b.sequence_order)
    for selection_a, selection_b in zip(plan_a.selections, plan_b.selections, strict=True):
        assert np.array_equal(selection_a.anchor_indices, selection_b.anchor_indices)
        assert np.array_equal(selection_a.target_indices, selection_b.target_indices)
        assert np.allclose(selection_a.transition_days, selection_b.transition_days)
        assert np.allclose(selection_a.transition_time_norm, selection_b.transition_time_norm)

    assert any(
        not np.array_equal(selection_a.anchor_indices, selection_c.anchor_indices)
        for selection_a, selection_c in zip(plan_a.selections, plan_c.selections, strict=True)
    )


def test_iter_resampled_pair_batches_loads_cpu_shards_incrementally(tmp_path: Path) -> None:
    """Iterator batches should gather only the sampled pairs from one shard at a time."""
    processed_dir = tmp_path / "processed"
    train_dir = processed_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    states0 = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    states1 = (100.0 + np.arange(4 * 3 * 2 * 2, dtype=np.float32)).reshape(4, 3, 2, 2)
    params0 = np.arange(7, dtype=np.float32)
    params1 = np.arange(7, dtype=np.float32) + 10.0
    np.savez(train_dir / "sim_000000.npz", states_norm=states0, params_norm=params0)
    np.savez(train_dir / "sim_000001.npz", states_norm=states1, params_norm=params1)

    shard_entries = [
        {"file": str(Path("train") / "sim_000000.npz")},
        {"file": str(Path("train") / "sim_000001.npz")},
    ]
    plan = ResampledSplitPlan(
        sequence_order=np.asarray([1, 0], dtype=np.int64),
        selections=(
            SequencePairSelection(
                anchor_indices=np.asarray([0, 1], dtype=np.int64),
                target_indices=np.asarray([1, 3], dtype=np.int64),
                transition_days=np.asarray([0.1, 0.3], dtype=np.float64),
                transition_time_norm=np.asarray([[-1.0], [0.5]], dtype=np.float32),
            ),
            SequencePairSelection(
                anchor_indices=np.asarray([1], dtype=np.int64),
                target_indices=np.asarray([3], dtype=np.int64),
                transition_days=np.asarray([0.2], dtype=np.float64),
                transition_time_norm=np.asarray([[1.5]], dtype=np.float32),
            ),
        ),
    )

    batches = list(
        _iter_resampled_pair_batches(
            processed_dir=processed_dir,
            shard_entries=shard_entries,
            plan=plan,
            batch_size=2,
            device=torch.device("cpu"),
        )
    )

    assert len(batches) == 2

    conditioning0, state_input0, state_target0 = batches[0]
    assert conditioning0.shape == (1, 8)
    assert torch.allclose(conditioning0[0, :-1], torch.from_numpy(params1))
    assert conditioning0[0, -1].item() == pytest.approx(1.5)
    assert torch.allclose(state_input0[0], torch.from_numpy(states1[1]))
    assert torch.allclose(state_target0[0], torch.from_numpy(states1[3]))

    conditioning1, state_input1, state_target1 = batches[1]
    assert conditioning1.shape == (2, 8)
    assert torch.allclose(conditioning1[:, :-1], torch.from_numpy(np.repeat(params0[None, :], 2, axis=0)))
    assert torch.allclose(conditioning1[:, -1], torch.tensor([-1.0, 0.5], dtype=torch.float32))
    assert torch.allclose(state_input1[0], torch.from_numpy(states0[0]))
    assert torch.allclose(state_input1[1], torch.from_numpy(states0[1]))
    assert torch.allclose(state_target1[0], torch.from_numpy(states0[1]))
    assert torch.allclose(state_target1[1], torch.from_numpy(states0[3]))


def test_iter_resampled_pair_batches_preloaded_gathers_from_gpu_split() -> None:
    """Preloaded resampled iteration should gather the same pairs from one resident split."""
    states0 = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    states1 = (100.0 + np.arange(4 * 3 * 2 * 2, dtype=np.float32)).reshape(4, 3, 2, 2)
    params0 = np.arange(7, dtype=np.float32)
    params1 = np.arange(7, dtype=np.float32) + 10.0
    split = PreloadedSequenceSplit(
        states=torch.from_numpy(np.stack([states0, states1], axis=0)),
        params=torch.from_numpy(np.stack([params0, params1], axis=0)),
    )
    plan = ResampledSplitPlan(
        sequence_order=np.asarray([1, 0], dtype=np.int64),
        selections=(
            SequencePairSelection(
                anchor_indices=np.asarray([0, 1], dtype=np.int64),
                target_indices=np.asarray([1, 3], dtype=np.int64),
                transition_days=np.asarray([0.1, 0.3], dtype=np.float64),
                transition_time_norm=np.asarray([[-1.0], [0.5]], dtype=np.float32),
            ),
            SequencePairSelection(
                anchor_indices=np.asarray([1], dtype=np.int64),
                target_indices=np.asarray([3], dtype=np.int64),
                transition_days=np.asarray([0.2], dtype=np.float64),
                transition_time_norm=np.asarray([[1.5]], dtype=np.float32),
            ),
        ),
    )

    batches = list(
        _iter_resampled_pair_batches_preloaded(
            split=split,
            plan=plan,
            batch_size=2,
        )
    )

    assert len(batches) == 2

    conditioning0, state_input0, state_target0 = batches[0]
    assert conditioning0.shape == (1, 8)
    assert torch.allclose(conditioning0[0, :-1], torch.from_numpy(params1))
    assert conditioning0[0, -1].item() == pytest.approx(1.5)
    assert torch.allclose(state_input0[0], torch.from_numpy(states1[1]))
    assert torch.allclose(state_target0[0], torch.from_numpy(states1[3]))

    conditioning1, state_input1, state_target1 = batches[1]
    assert conditioning1.shape == (2, 8)
    assert torch.allclose(conditioning1[:, :-1], torch.from_numpy(np.repeat(params0[None, :], 2, axis=0)))
    assert torch.allclose(conditioning1[:, -1], torch.tensor([-1.0, 0.5], dtype=torch.float32))
    assert torch.allclose(state_input1[0], torch.from_numpy(states0[0]))
    assert torch.allclose(state_input1[1], torch.from_numpy(states0[1]))
    assert torch.allclose(state_target1[0], torch.from_numpy(states0[1]))
    assert torch.allclose(state_target1[1], torch.from_numpy(states0[3]))
