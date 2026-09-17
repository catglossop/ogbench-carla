"""Single source of truth for which Gemini model the coaches use.

Before this module the model name was hard-coded in a dozen places with THREE different
values -- ``gemini-2.0-flash`` in ``vlm_feedback`` and ``main_carla``'s VLM-coach path,
``gemini-3.5-flash`` in ``cast_relabel`` / ``static_coach`` / ``online_static_coach`` and every
agent config -- so which model a run actually used depended on which entry point built the coach.
Two runs could differ in their reviewer and agree in every logged config field.

Change the model HERE. Anything that needs a different one must pass it explicitly.
"""

# The model every coach uses unless a caller overrides it.
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"

# Runs launched with --eval-mode are the ones whose numbers get reported, so they must not
# silently inherit an older model from a stale config. main_carla forces this for them.
EVAL_MODE_GEMINI_MODEL = DEFAULT_GEMINI_MODEL
