## Implicit Safety Alignment from Crowd Preferences

[Paper](https://arxiv.org/abs/2605.21822)

This repository contains the code used to run the experiments reported in the paper **Implicit Safety Alignment from Crowd Preferences**, accepted at ICML 2026.
This codebase builds upon the LLM experiments of Variational Preference Learning (VPL). In our paper, we use this repository to evaluate whether safety-aligned preference information learned from crowd preferences can be transferred to downstream language-based decision tasks.

**Branch Information**
- `main` branch: main experiments on Safe RL environments
- `llm` branch: LLM-style experiments and evaluation

---

## Instructions

#### Install Dependencies

```bash
conda create -n SafeCrowdPref-LLM python=3.10
conda activate SafeCrowdPref-LLM
pip install -r requirements.txt
```

---

## Full Experimental Pipeline

### 1. Generate Crowd Preference Data

```bash
bash generate_llm_embeddings_pets.sh gpt2
```

---

### 2. Train Reward Models

Train the VAE-based reward model and the unimodal baseline reward model:

```bash
bash submit_job_pets.sh vae
bash submit_job_pets.sh base
```

---

### 3. Evaluate Downstream Policies

To evaluate the baseline downstream policy:

```bash
train_bandit_policy.ipynb
```

To evaluate our method based on latent skill composition:

```bash
train_vae_bandit_policy.ipynb
```

---

## Acknowledgement

This repository builds upon the LLM experiment codebase from Variational Preference Learning (VPL): https://github.com/WEIRDLabUW/vpl_llm

---

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{lin2026implicit,
  title={Implicit Safety Alignment from Crowd Preferences},
  author={Lin, Qian and Brown, Daniel S.},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```