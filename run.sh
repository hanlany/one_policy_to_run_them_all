touch bash_history
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.Xauthority \
    -v $HOME/.Xauthority:/tmp/.Xauthority:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./bash_history:/root/.bash_history \
    --network=host \
    --gpus all \
    -v ./../one_policy_to_run_them_all:/app/one_policy_to_run_them_all \
    lavaurma \
    bash



