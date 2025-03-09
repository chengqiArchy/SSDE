# Structured Sparsity and Dormant-guided Exploration (SSDE)

## Overview
This repository contains the implementation of the paper [Mastering Continual Reinforcement Learning through Fine-grained Network Allocation and Dormant Neuron Exploration](https://openreview.net/pdf?id=3ENBquM4b4). Our method addresses the continual learning problem in Meta-World by leveraging a structure-based approach to mitigate catastrophic forgetting. Additionally, we introduce a new metric, input sensitivity, to measure neuron dormancy, which helps tackle the challenging and unstable stick-pull sub-task.

## Key Dependencies

To create a new environemnt:

```bash
conda create -name dev_co python==3.10.14
conda activate dev_co
```

Here are some key dependencies:
```bash
jax==0.4.32
jax-cuda12-pjrt==0.4.32
jax-cuda12-plugin==0.4.32
jaxlib==0.4.32
jaxopt==0.8.3
flax==0.9.0
numpy==1.26.4
```

You can simply run the code here:
```bash
pip install -r requirements.txt
```



Our main results are based on Metaworld v1, the same version used in [Continual World](https://github.com/awarelab/continual_world)

To set up the environment, follow these instructions, (make sure you are in the same conda environment):
```bash
git clone https://github.com/Farama-Foundation/Metaworld
cd Metaworld
git checkout 0875192
pip install -e .
```


## Quick Start

You can simply run the bash script here:

```bash
./run.sh
```

> Here a an link to reproduced [CoTASP_result](https://wandb.ai/iclr_2025_ssde_continual_rl-iclr/CoTASP_Testing/reports/ICLR2025_CoTASP_Reproduce--VmlldzoxMDM1MzAzNg?accessToken=22xe9avpmoynbchfcwyg4utxqsypxsre5r4yxamyfs3wcstsajc0ygjq0hzats3t), the setting have been asligned with the original `continual world` setting.

## Acknowledgement

We appreciate the valuable work of the following repositories, which greatly contributed to our implementation:

- [continual_world](https://github.com/awarelab/continual_world) – Provided an important benchmark for continual learning.
- [Metaworld](https://github.com/Farama-Foundation/Metaworld) – Used as the core environment for our experiments.
- [CoTASP](https://github.com/stevenyangyj/CoTASP.git) – the base code of this implementation.

Special thanks to the open-source community for their continuous contributions!

