# Towards Generalizable Ultrasound Segmentation via Physics-Guided Supervision

This repository contains the official implementation of our paper "Towards Generalizable Ultrasound Segmentation via Physics-Guided Supervision".

Authors: Ghafoor Haris, Minwoo Shin & Kyungho Yoon
## Summary

We propose a physics-guided ultrasound segmentation framework that simulates B-mode images by solving the acoustic wave equation and reduces the simulation-to-real gap using style-preserving adaptation. The resulting models generalize well across diverse clinical datasets and transfer effectively to downstream tasks.
This repository provides the simulation pipeline, style transfer, and segmentation training and evaluation code.

![Method overview](assets/method.png)

## Repository Structure

- `physics_guided_simulation/`: Physics-based ultrasound simulation pipeline (k-Wave).
- `phyusnet/`: Segmentation training, fine-tuning, and evaluation.
- `style_augmentation/`: Style transfer with an APSA-based novel loss; based on the original CycleGAN repo for reference.
- `benchmark_datasets/US30K/`: Clinical datasets organized by task.
- `assets/`: Paper figures used in this README.

## Installation

```bash
pip install -r physics_guided_simulation/requirements.txt
```

## Data

### Physics Simulation Output

The simulation pipeline generates B-mode images and corresponding labels. Outputs are referenced via an `.npz` file with `train`, `val`, and `test` splits, where each entry stores `(scan_path, label_path, sample_id)`. Our dataset can be accessed via Zenodo repository [link](https://zenodo.org/records/18449815?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRjOGYwYjkyLWIzZDMtNDA5ZS1iNjBkLWU4NmU4NTFlNDU4MiIsImRhdGEiOnt9LCJyYW5kb20iOiJhOWNlZTlkOTA1NGUwNTY5MDc4YmEzNTNiMjA5NTUzNyJ9.0EIZyQEuAddLju1JOAiqTX25Wh2_EdO19hTbLFP__GiTmWf8NRMhAV-ywSIVW8QRWTEtc1v5lz56xGiLpYNJVw)

### Clinical Datasets

Clinical datasets are organized under `benchmark_datasets/US30K/`, each with `img/` and `label/` subfolders. Download the datasets and train-test split information from the Google Drive folder [link](https://drive.google.com/file/d/1P9VYaNEq8ww42c2lsAKa0C2gqMjPP6UJ/view?usp=drive_link).
Supported datasets include BUSI, UDIAT, TN3K, DDTI, MMOTU2D, and others.

## Usage

### 1. Physics-Guided Simulation

```bash
cd physics_guided_simulation
python run_python_pipeline.py --end-sample-id 100 --gpu --noise-levels low medium high
```

### 2. Style Augmentation (Synthetic-to-Real)

```bash
cd style_augmentation
python train.py
python test.py
```

Checkpoints for synthetic-to-real ultrasound transfer models are available in the Google Drive folder [link]().

### 3. Physics Pretraining

```bash
cd phyusnet
python main.py --dataset_path /path/to/physics_data/paths.npz --model_name segformer --epochs 100
```

All segmentation checkpoints are available in the HuggingFace [link](https://huggingface.co/haris2k/PhyUS-Net).
### 4. Task-Specific Fine-Tuning

```bash
cd phyusnet
python task_specific.py --mode finetune --dataset_key BUSI --model_name segformer --epochs 50
```

### 5. Evaluation

```bash
cd phyusnet
python utils/test.py --benchmark_dataset BUSI --model_name segformer --task physics --checkpoint_folder_path ./checkpoints
```

## Results

![Main results](assets/results.png)

Quantitative results report Dice, IoU, and HD95 across multiple datasets in the US30K benchmark, highlighting zero-shot generalization and transfer to downstream tasks.

## Checkpoints

- Physics-pretrained models: `phyusnet/checkpoints/physics_models/`
- Task-specific fine-tuned models: `phyusnet/checkpoints/task_specific_models/`
All checkpoints are also available in HuggingFace [link](https://huggingface.co/haris2k/PhyUS-Net).

## Citation

If you use this codebase or dataset, please cite:

@misc{ghafoor2026physicsguided,
  author       = {Ghafoor, Haris and Shin, Minho and Kyungho, Yoon},
  title        = {Towards Generalizable Ultrasound Segmentation via Physics-Guided Supervision},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18449815},
  url          = {https://doi.org/10.5281/zenodo.18449815}
}


@misc{ghafoor_2026,
	author       = { ghafoor },
	title        = { PhyUS-Net (Revision b5279a7) },
	year         = 2026,
	url          = { https://huggingface.co/haris2k/PhyUS-Net },
	doi          = { 10.57967/hf/7695 },
	publisher    = { Hugging Face }
}



## Acknowledgements

The work was supported by the National Research Foundation of Korea (NRF) funded by the Korean government (MSIT) under Grants RS-2024-00335185. This work was also supported by the Regional Innovation System \& Education(RISE) program through the Gangwon RISE Center, funded by the Ministry of Education (MOE) and the Gangwon State (G.S.), Republic of Korea (2025-RISE-10-006). 

We acknowledge the developers of k-Wave and Python k Wave, as well as the authors of the open source implementations of U Net, U Net plus plus, SegFormer, Swin U Net, and Trans U Net. We also acknowledge the maintainers of segmentation models pytorch and the CycleGAN and pix2pix codebases used for style augmentation.
## License

This project is licensed under the MIT License. See the LICENSE file for details.

