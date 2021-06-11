
# BoolNet: Minimizing The Energy Consumption of Binary Neural Networks 

This is the accompanying code for our paper [BoolNet: Minimizing The Energy Consumption of Binary Neural Networks](todo).

## Setup

In any case need to have a [CUDA setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
with a recent NVIDIA driver on your host system.
(We tested driver version 465.27 with support for CUDA 11.3, but drivers supporting CUDA 11.1 should be fine.)

Further you need to download and prepare the ImageNet dataset ([how?](https://stackoverflow.com/a/62253211)).

We use [enroot](#with-enroot) to provide an independent environment,
but you can install it [directly on your machine](#direct-install) instead.

### With Enroot
*(Last tested on: 2021-06-11)*

Install [enroot](https://github.com/NVIDIA/enroot) and
[libnvidia-container](https://github.com/nvidia/libnvidia-container) (for GPU support).

1. Import pytorch docker image into enroot and install the requirements.
```bash
enroot import -o pytorch.sqsh docker://pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
enroot create pytorch.sqsh
enroot start --root --rw -m ".":"/workspace" -- pytorch ./install.sh
```
2. (Optional) Repack the modified image (to have a clean image stored):
```bash
enroot export --output boolnet.sqsh pytorch
enroot remove pytorch && rm pytorch.sqsh
enroot create boolnet.sqsh
```
3. Select the experiment and run the corresponding training command (replace `boolnet` with `pytorch` if you did not do the previous step):
```bash
cd BaseNet_k_1
cd src
enroot start --rw -m "/path/to/imagenet/data":"/mnt/imagenet" -m ".":"/workspace" -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=x -- boolnet ./run.sh
```

### Direct Install

Please check our script `./install.sh` (designed for Ubuntu systems) to install the requirements needed
(in addition to the previously mentioned CUDA installation).
Please adapt the script accordingly for other operating systems (or use enroot instead).

After you have installed all requirements, simply copy and run the training command and add `--imagenet_directory /path/to/imagenet`:
```bash
cd BaseNet_k_1
cd src
cat run.sh # print the training command for copying
# paste the command, add the imagenet directory (--imagenet_directory ...) and run 
```

## Experiment details

We provide more details about our individual runs (log files, accuracy curves, etc.),
in the corresponding directories:

- [BaseNet(k=1)](BaseNet_k=1)
- [BaseNet(k=4)](BaseNet_k=4)
- [BoolNet(k=4)](BoolNet_k=4)
- [BoolNet*(k=4)](BoolNet_k=4_star)

## Contributing

Please feel free to open an issue or a pull request, if you encounter problems or want to provide suggestions.

## Cite

If you want to compare to our results or if our code has helped your research, we would be happy if you can cite us:
```
TODO
```
