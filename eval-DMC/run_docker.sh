    docker run --rm --gpus all -it \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v $(pwd):/workspace/eai-vc \
    --shm-size=16gb \
    [your docker image name]