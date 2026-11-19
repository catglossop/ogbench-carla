"""A KV-cached replacement for SimLingo's ``LLM.greedy_sample``.

The stock loop (simlingo_training/models/language_model/llm.py) re-runs the whole sequence through
the language model for EVERY generated token: it appends the sampled token's embedding to
``input_embeds`` and calls ``forward`` on the lot again, with no ``past_key_values``. With
``max_new_tokens=512`` and a prompt that already holds hundreds of image tokens, almost all of that
work is recomputing a prefix that cannot change.

This version keeps the cache instead, so each step feeds one token and attends over the stored keys
and values. It is arithmetically the same computation, and is written to keep the stock loop's
observable behaviour exactly:

  * logits come from ``F.linear(last_hidden_state, logit_matrix)``, not the model's own head;
  * ``sample_categorical`` is called once per step with the same arguments, so the RNG is consumed
    in the same order and a seeded run draws the same tokens;
  * ``sampled_tokens`` is pre-filled with eos, written only for rows still incomplete, and trimmed
    on the step where every row has finished;
  * the second return value is the full prompt-plus-generated embedding sequence, as before.

Patched in here rather than in the simlingo checkout: that tree is shared with the SimLingo
residual-SAC stack, and the HL worker already replaces this method to set the sampling temperature.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def install_kv_cache(hl, sampling: dict):
    """Replace ``hl.model.language_model.greedy_sample`` with the cached loop.

    ``sampling`` is the worker's live dict; ``sampling['temperature']`` is read per call, exactly as
    the temperature wrapper it supersedes. Returns the original bound method so a caller can compare
    the two or restore it.
    """
    llm = hl.model.language_model
    original = llm.greedy_sample

    @torch.no_grad()
    def cached_greedy_sample(
        input_embeds,
        inputs_mask=None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
        cache_offset: int = 0,
        input_embed_matrix=None,
        logit_matrix=None,
        restrict_tokens=None,
        attention_mask=None,
        position_ids=None,
    ):
        temperature = float(sampling.get('temperature', temperature))
        if input_embed_matrix is None:
            if llm.embed_tokens is None:
                raise ValueError('No input embeddings available; pass input_embed_matrix.')
            input_embed_matrix = llm.embed_tokens.weight
        if logit_matrix is None:
            if llm.lm_head is None:
                raise ValueError('No logit matrix available; pass logit_matrix.')
            logit_matrix = llm.lm_head.weight

        rows = input_embeds.size(0)
        sampled_tokens = torch.empty((rows, max_new_tokens), device=input_embeds.device, dtype=torch.long)
        if eos_token_id is not None:
            sampled_tokens.fill_(eos_token_id)
        incomplete_seq_mask = torch.ones(rows, dtype=torch.bool, device=input_embeds.device)

        # Only the new token is fed once the cache exists; attention_mask still spans the whole
        # sequence (cache included), which is what the model expects.
        step_embeds = input_embeds
        appended = []
        past = None
        for i in range(max_new_tokens):
            outputs = llm.model(
                inputs_embeds=step_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden_state = outputs.hidden_states[-1][:, -1]
            logits = F.linear(last_hidden_state, logit_matrix)
            next_token = llm.sample_categorical(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, restrict_tokens=restrict_tokens
            )
            step_embeds = F.embedding(next_token.unsqueeze(1), input_embed_matrix)
            appended.append(step_embeds)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((rows, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                    dim=1,
                )
            sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
            if eos_token_id is not None:
                incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
                if not incomplete_seq_mask.any():
                    sampled_tokens = sampled_tokens[:, : i + 1]
                    break

        return sampled_tokens, torch.cat([input_embeds, *appended], dim=1)

    llm.greedy_sample = cached_greedy_sample
    return original
