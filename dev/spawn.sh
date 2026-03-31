touch bash_history
xhost +local:docker
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.Xauthority \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e MESA_LOADER_DRIVER_OVERRIDE=swrast \
    -e MUJOCO_GL=egl \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e PYOPENGL_PLATFORM=egl \
    -v $HOME/.Xauthority:/tmp/.Xauthority:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./bash_history:/root/.bash_history \
    --name=lavaurma_dev \
    --network=host \
    --gpus all \
    -v ./../one_policy_to_run_them_all:/app/one_policy_to_run_them_all \
    -v ./../RL-X:/app/RL-X \
    -v ./../GenLoco:/app/GenLoco \
    -v ./../genloco-loihi:/app/genloco-loihi \
    lavaurma \
    bash

