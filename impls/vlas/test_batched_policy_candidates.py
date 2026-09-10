"""Focused CPU-only checks for the guarded Qwen policy-candidate batch."""

from __future__ import annotations

import jax
import numpy as np
import ast
from pathlib import Path
from types import SimpleNamespace

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
        "reasoning_texts": [f"<loc1020>Follow the route. Decelerate for vehicle {i}.;<loc1019>" for i in range(n)],
        "reasoning_overflowed": np.zeros(n, dtype=bool),
    }


def _sample(actor, label_source="subtask"):
    return sample_batched_policy_candidates(
        actor=actor,
        raw={"image": np.zeros((4, 4, 3), dtype=np.uint8)},
        rng=jax.random.PRNGKey(7),
        num_candidates=8,
        model_noise_dim=320,
        env_action_dim=40,
        noise_scale=1.25,
        label_source=label_source,
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


def test_commentary_uses_reasoning_without_changing_actions_or_noise():
    legacy = _FakeActor(_valid_result())
    corrected = _FakeActor(_valid_result())
    old_actions, _ = _sample(legacy)
    actions, labels = _sample(corrected, "commentary")
    assert labels == [f"Follow the route. Decelerate for vehicle {i}." for i in range(8)]
    np.testing.assert_array_equal(actions, old_actions)
    np.testing.assert_array_equal(corrected.calls[0][1]["noise"], legacy.calls[0][1]["noise"])
    assert actions.shape == (8, 40)


def test_missing_commentary_never_falls_back_to_summary():
    for bad in (None, [], ["<loc1020>;<loc1019>"] * 8, [None] * 8):
        result = _valid_result()
        result["reasoning_texts"] = bad
        try:
            _sample(_FakeActor(result), "commentary")
        except BatchedCandidateValidationError as exc:
            assert "commentary" in str(exc)
        else:
            raise AssertionError("Missing commentary must not silently use subtask_texts")


def test_refined_labels_remove_actor_markers_without_changing_actions():
    result = _valid_result()
    result['subtask_texts'] = [f'<loc1022>The vehicle accelerates for candidate {i}.;<loc1021>' for i in range(8)]
    actions, labels = _sample(_FakeActor(result), 'subtask')
    assert labels == [f'The vehicle accelerates for candidate {i}.' for i in range(8)]
    expected_actions, _ = _sample(_FakeActor(_valid_result()), 'subtask')
    np.testing.assert_array_equal(actions, expected_actions)


def test_sequential_qwen_path_uses_matching_commentary():
    # Execute the actual nested sampler without importing main_carla's simulator
    # setup. Its actor/agent stand-ins make candidate-to-label pairing observable.
    source = Path(__file__).resolve().parents[1] / "main_carla.py"
    sampler = next(n for n in ast.walk(ast.parse(source.read_text()))
                   if isinstance(n, ast.FunctionDef) and n.name == "_sample_diverse_candidates")
    state = {"sample": 0}

    def sample_action(obs, noise):
        state["sample"] += 1
        assert noise.shape == (1, 320)
        return jax.numpy.full((1, 40), state["sample"], dtype=jax.numpy.float32)

    actor = SimpleNamespace(decode_last_batch_reasoning=lambda: [
        f"<loc1020>Follow the route. Candidate {state['sample']}.;<loc1019>"
    ])
    agent = SimpleNamespace(vla_sample_fn=sample_action, _flat_noise_dim=lambda: 320,
                            _clip_actions_to_env=lambda x: x)
    scope = dict(jax=jax, np=np, FLAGS=SimpleNamespace(
        bon_max_sample_attempts=1, bon_batch_policy_candidates=False,
        bon_qwen_label_source="commentary"), _qwen_selector=object(),
        steervla_actor=actor, agent=agent, obs=np.zeros(1), _vla_noise_scale=1.25)
    exec(compile(ast.Module(body=[sampler], type_ignores=[]), str(source), "exec"), scope)
    actions, labels = scope["_sample_diverse_candidates"](jax.random.PRNGKey(7), 2)
    assert labels == ["Follow the route. Candidate 1.", "Follow the route. Candidate 2."]
    np.testing.assert_array_equal(actions, np.repeat([[1.], [2.]], 40, axis=1))


if __name__ == "__main__":
    test_one_batch_preserves_noise_and_normalized_action_contract()
    test_physical_only_result_is_rejected()
    test_overflowed_cot_is_rejected()
    test_commentary_uses_reasoning_without_changing_actions_or_noise()
    test_missing_commentary_never_falls_back_to_summary()
    test_sequential_qwen_path_uses_matching_commentary()
    print("[ok] guarded batched Qwen policy-candidate sampling")


def test_batched_cot_reuse_uses_env_steps_and_resets_on_episode_change():
    result = _valid_result()
    result['cot_out'] = {'tokens': np.arange(8)}
    actor = _FakeActor(result)
    actor.actions_per_cot = 5
    def query(step, episode=0):
        return sample_batched_policy_candidates(
            actor=actor, raw={'routing_command': 'Turn left'},
            rng=jax.random.PRNGKey(step), num_candidates=8,
            model_noise_dim=320, env_action_dim=40, noise_scale=1.0,
            episode_index=episode, episode_step=step,
        )
    query(0)
    query(3)
    query(6)
    query(0, episode=1)
    assert 'cot_out' not in actor.calls[0][1]
    assert actor.calls[1][1]['cot_out'] is result['cot_out']
    assert 'cot_out' not in actor.calls[2][1]
    assert 'cot_out' not in actor.calls[3][1]
    assert len(actor.calls) == 4  # Reusing text never reuses action chunks.
    assert not np.array_equal(actor.calls[0][1]['noise'], actor.calls[1][1]['noise'])
