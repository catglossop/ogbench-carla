```
./run_carla.sh --train-gpu 2 --sim-gpu 3 --carla-port 2020 --carla-streaming-port 2021 --tm-port 8020 --x-display-num 30
```

```
./run_carla.sh --train-gpu 0 --sim-gpu 4 --carla-port 2030 --carla-streaming-port 2031 --tm-port 8030 --x-display-num 31
```

pkill -f 'Xvfb :30'
pkill -f 'CarlaUE4.*12020'

./run_carla.sh \
    --train-gpu 0 \
    --render-adapter 3 \
    --carla-port 12067 \
    --carla-streaming-port 12061 \
    --tm-port 18010 \
    --x-display-num 32


    