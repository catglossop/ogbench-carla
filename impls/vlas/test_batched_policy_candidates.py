"""Focused CPU-only checks for the guarded Qwen policy-candidate batch."""

from __future__ import annotations

import jax
import numpy as np

from vlas.batched_candidates import (
    BatchedCandidateValidationError,
    sample_batched_policy_candidates,
)


class _FakeActor:
    cot_temperature = 0.75

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.reset_calls = 0

    def reset_action_cache(self):
        self.reset_calls += 1

    def sample_candidates(self, n, **kwargs):
        self.calls.append((n, kwargs))
        return self.result


def _valid_result(n=8, width=40):
    return {
        "actions": np.full((n, width), 999.0, dtype=np.float32),
        "actions_normalized": np.arange(n * width, dtype=np.float32).reshape(n, width),
        "subtask_texts": [f"candidate {i}" for i in range(n)],
        "reasoning_overflowed": np.zeros(n, dtype=bool),
    }


def _sample(actor):
    return sample_batched_policy_candidates(
        actor=actor,
        raw={"image": np.zeros((4, 4, 3), dtype=np.uint8)},
        rng=jax.random.PRNGKey(7),
        num_candidates=8,
        model_noise_dim=320,
        env_action_dim=40,
        noise_scale=1.25,
    )


def test_one_batch_preserves_noise_and_normalized_action_contract():
    actor = _FakeActor(_valid_result())
    actions, subtasks = _sample(actor)

    assert actor.reset_calls == 1
    assert len(actor.calls) == 1
    n, kwargs = actor.calls[0]
    assert n == 8
    assert kwargs["temperature"] == actor.cot_temperature
    noise = np.asarray(kwargs["noise"])
    assert noise.shape == (8, 320)
    assert np.isfinite(noise).all()
    assert np.abs(noise).max() <= 1.25
    assert np.unique(noise, axis=0).shape[0] == 8
    assert np.array_equal(actions, actor.result["actions_normalized"])
    assert not np.array_equal(actions, actor.result["actions"])
    assert subtasks == actor.result["subtask_texts"]


def test_physical_only_result_is_rejected():
    result = _valid_result()
    del result["actions_normalized"]
    try:
        _sample(_FakeActor(result))
    except BatchedCandidateValidationError as exc:
        assert "physical-unit" in str(exc)
    else:
        raise AssertionError("physical-only result should be rejected")


def test_overflowed_cot_is_rejected():
    result = _valid_result()
    result["reasoning_overflowed"][3] = True
    try:
        _sample(_FakeActor(result))
    except BatchedCandidateValidationError as exc:
        assert "row" in str(exc) and "3" in str(exc)
    else:
        raise AssertionError("overflowed CoT should be rejected")


if __name__ == "__main__":
    test_one_batch_preserves_noise_and_normalized_action_contract()
    test_physical_only_result_is_rejected()
    test_overflowed_cot_is_rejected()
    print("[ok] guarded batched Qwen policy-candidate sampling")
