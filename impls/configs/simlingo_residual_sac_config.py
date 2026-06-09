"""Default config for SimLingo + Residual SAC on CARLA Bench2Drive.

Reference these values when running main_carla_simlingo.py.
The flags in that script override these values; this file is for documentation.
"""

SIMLINGO_CHECKPOINT = (
    "/home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt"
)

# Residual SAC
RES_SCALE = 0.1          # final = base + 0.1 * residual  (kept small initially)
GAMMA = 0.97
TAU = 0.01               # target network soft-update
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
TARGET_ENTROPY = -2.0    # = -action_dim

# Architecture
VLM_FEATURE_DIM = 896    # InternVL2-1B Qwen2 backbone (30 driving tokens, mean-pooled)
ACTOR_HIDDEN_DIMS = (256, 256, 256)

# Training
BATCH_SIZE = 256
BUFFER_CAPACITY = 10_000
LEARNING_STARTS = 500    # steps before SAC updates begin
WARMUP_STEPS = 500       # steps using zero residual (base policy only)
UPDATES_PER_STEP = 10    # gradient steps per env step / UTD ratio

# Carla leaderboard agent
SIMLINGO_AGENT = "ogbench/carla/leaderboard_agents/simlingo_obs.py"
CARLA_CONFIG = "impls/configs/carla_config.yaml"
