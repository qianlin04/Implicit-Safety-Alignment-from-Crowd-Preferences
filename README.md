## Implicit Safety Alignment from Crowd Preferences


This is an implementation of Paper: Implicit Safety Alignment from Crowd Preferences.

## Instructions


#### Install Dependencies
```
conda create -n SafeCrowdPref python=3.10
conda activate SafeCrowdPref
pip install -r requirements.txt
pip install -e dependencies/d4rl --no-deps
pip install -e dependencies/Bullet-Safety-Gym
pip install -e .
Install "mujoco-py" from https://github.com/openai/mujoco-py
```


## Data Generation

To collect offline datasets using SAC and generate crowd preference datasets, run:

```
bash scripts/data_generation.sh
```


## Skill Discover and Downstream Training

To perform skill discovery using the original VPL method and then train downstream policies, run:

```
bash scripts/train.sh 0 VAE
```

To use the CPL-based variant for skill discovery instead, run:

```
bash scripts/train.sh 0 VAEPolicy
```



## Acknowledgement

This repository builds upon the original VPL codebase from
https://github.com/WEIRDLabUW/vpl
.

We use the IQL implementation from https://github.com/gwthomas/IQL-PyTorch, and the TD3 / TD3+BC implementations from https://github.com/sfujim/TD3 and https://github.com/sfujim/TD3_BC.


## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{lin2026implicit,
  title={Implicit Safety Alignment from Crowd Preferences},
  author={Lin, Qian and Brown, Daniel S.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```