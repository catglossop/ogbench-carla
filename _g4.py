p = 'run_carla.sh'
s = open(p).read()

# defaults: empty means "not set", so main_carla's own fallbacks to --seed apply
old = 'EVAL_MODE="false"\n'
assert s.count(old) == 1
s = s.replace(old, old + 'CARLA_SEED=""\nTRAIN_SEED=""\nEVAL_SEEDS=""\n', 1)

# parsing, next to --seed
old = '    --seed) SEED="$2"; shift 2 ;;\n'
assert s.count(old) == 1
s = s.replace(
    old,
    old
    + '    --carla-seed|--carla_seed) CARLA_SEED="$2"; shift 2 ;;\n'
    + '    --train-seed|--train_seed) TRAIN_SEED="$2"; shift 2 ;;\n'
    + '    --eval-seeds|--eval_seeds) EVAL_SEEDS="$2"; shift 2 ;;\n',
    1,
)

# forward only when set, so the unset case keeps main_carla's --seed fallback
old = '    --seed="${SEED}" \\\n'
assert s.count(old) == 1
s = s.replace(
    old,
    old
    + '    ${CARLA_SEED:+--carla_seed="${CARLA_SEED}"} \\\n'
    + '    ${TRAIN_SEED:+--train_seed="${TRAIN_SEED}"} \\\n'
    + '    ${EVAL_SEEDS:+--eval_seeds="${EVAL_SEEDS}"} \\\n',
    1,
)

# help, next to --seed's line
old = '  --seed N                  Random seed. Default: 0\n'
assert s.count(old) == 1
s = s.replace(
    old,
    old
    + '  --carla-seed N            Simulator-side seed: traffic manager, scenario actors,\n'
    + '                            env.reset(). Held fixed through the --eval-mode eval\n'
    + '                            episodes. Default: --seed.\n'
    + '  --train-seed N            Model-side seed: JAX PRNG, numpy/random, agent construction,\n'
    + '                            and the actor\'s CoT / action / noise sampling. Default: --seed.\n'
    + '  --eval-seeds A,B,C        Comma-separated MODEL seeds for the --eval-mode eval episodes\n'
    + '                            (no spaces). Each replays the same --carla-seed, so their\n'
    + '                            spread measures the policy, not the scenario.\n'
    + '                            Default: train-seed+1001, +1002, ... one per eval episode.\n',
    1,
)

open(p, 'w').write(s)
print("run_carla.sh: --carla-seed / --train-seed / --eval-seeds added")
