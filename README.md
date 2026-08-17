# Complementary CFA and JDENet: Co-Design of CMYW CFA and Joint Demosaicking-Enhancement Network for Low-Light Imaging

Citation
If you find this work useful for your research, please consider citing our paper:

@article{jiang2026complementary,
  title={Complementary CFA and JDENet: Co-Design of CMYW CFA and Joint Demosaicking-Enhancement Network for Low-Light Imaging},
  author={Jiang, Yunfeng and Huang, Fuyu and Li, Ruiqiang and Liu, Limin and Li, Zhaorui and Zheng, Chaowen and Tian, Kuo and Wang, Dan and Wu, Dongsheng},
  journal={Knowledge-Based Systems},
  year={2026},
  note={Under review}
}

Dataset Citation:
@dataset{jiang2026cmyw,
  title={Real CMYW Low-Light Image Enhancement Dataset},
  author={Jiang, Yunfeng and Wu, Dongsheng and others},
  year={2026},
  note={Available at: \url{https://pan.baidu.com/s/1M17r9g9meAIPgnSPSp-kMg?pwd=1111}, Extraction code: 1111}
}

#### Yunfeng Jiang, Fuyu Huang, Ruiqiang Li, Limin Liu, Zhaorui Li, Chaowen Zheng, Kuo Tian, Dan Wang, Dongsheng Wu*

> Low-light color imaging demands higher-sensitivity sensors. Traditional Bayer CFAs suffer from low light intake, and existing hand-crafted CFAs are designed separately from demosaicking and enhancement, disrupting the coupling between sampling and reconstruction and hindering end-to-end optimization. In this paper, we propose a co-design framework that, for the first time, jointly optimizes the configuration of a complementary CMYW (cyan, magenta, yellow, white) CFA and the parameters of a Joint Demosaicking-Enhancement Network (JDENet). We replace the RGGB pattern with CMYW to improve light efficiency and adopt an improved Gumbel-Softmax function to achieve differentiable channel selection. JDENet decouples processing into a high-sensitivity W detail branch and a spectrally guided CMY color branch, with spatial and channel attention mechanisms. The backbone uses SCERes2B within a U-Net architecture. In the deepest feature extraction stage, a Global and Local Multi-information Fusion Block (GLMIFB) integrates Transformer-based spatial and channel self-attention for global context and channel dependencies, along with a ConvNext-based local detail enhancement branch. Training employs L₂ and ΔE color loss for balanced reconstruction of detail and color. We also introduce the first publicly available real CMYW low-light image enhancement dataset, filling a gap in complementary-color imaging datasets. Extensive experiments on public and self-constructed datasets show our co-design significantly outperforms state-of-the-art methods in both quantitative evaluation and visual perception.

---

## Repository Structure
CMYW-CFA-JDENet/
├── train/
│ ├── train_learn_cmyw_to_rgb.py # Training on synthetic CMYW data
│ └── train_learn_cmyw_to_rgb_real.py # Training on real CMYW dataset
├── Test/
│ ├── Test_learn_cmyw_to_rgb.py # Testing on synthetic CMYW data
│ └── Test_learn_cmyw_to_rgb_real.py # Testing on real CMYW dataset
├── data/ # Dataset directory
├── nets/ # Network definitions (JDENet, SCERes2B, GLMIFB)
├── utils/ # Utility functions
├── shutters/ # CFA pattern definitions
├── snapshots/ # Model checkpoint saving directory
├── src/ # Additional source files
├── place2/ # Placeholder / auxiliary files
├── psnr_ssim/ # PSNR/SSIM evaluation scripts
├── JDENet.zip # Archived code package
└── README.md # This file


---

## Dataset

We release the first publicly available real CMYW low-light image enhancement dataset, comprising 516 registered scene pairs captured via a split-band acquisition system. The dataset includes:

- Short-exposure CMYW four-band low-light input images
- Long-exposure RGB ground-truth images
- Multiple illumination levels ranging from 1 to 10.00 lx

**Dataset Download**: [https://pan.baidu.com/s/1M17r9g9meAIPgnSPSp-kMg?pwd=1111](https://pan.baidu.com/s/1M17r9g9meAIPgnSPSp-kMg?pwd=1111) (Extraction code: 1111)

---

## Key Contributions

1. **Adaptive high-light-intake CMYW CFA design framework** — using improved Gumbel-Softmax reparameterization to overcome the non-differentiability of CFA channel selection, enabling end-to-end joint optimization with the reconstruction network.

2. **Joint Demosaicking and Enhancement Network (JDENet)** — a dual-branch network with high-sensitivity W detail branch and spectrally guided CMY color branch, integrating SCERes2B and Global and Local Multi-information Fusion Block (GLMIFB).

3. **First public real CMYW low-light enhancement dataset** — 516 registered scene pairs, filling a critical gap in complementary-color imaging resources.

4. **Balanced hybrid loss function** — combining L₂ and ΔE for simultaneous detail preservation and color fidelity.

---

## Code Running Guide

### Environment Installation

We use PyTorch for training and testing. Assuming you have [Anaconda](https://www.anaconda.com/products/individual#Downloads) installed:

```bash
conda create -n cmyw_jdenet python=3.8
conda activate cmyw_jdenet
pip install -r requirements.txt

### Data Preparation
Option 1: Synthetic CMYW Data (SIED Dataset)
We use the SIED dataset for synthetic experiments.

Download the SIED dataset.

Preprocess the RAW images to obtain 16-bit linear RGB images using RawPy.

Convert RGB images to CMYW four-channel images using the linear transformation:
C = G + B,  M = R + B,  Y = R + G,  W = R + G + B
Place the preprocessed data in the data/ directory.


Option 2: Real CMYW Dataset
Download our real CMYW dataset from: https://pan.baidu.com/s/1M17r9g9meAIPgnSPSp-kMg?pwd=1111 (Extraction code: 1111)

Place the dataset in the data/real_cmyw/ directory:
data/real_cmyw/
├── train/          # 466 pairs
└── test/           # 50 pairs

### Training
Train on Synthetic CMYW Data (SIED)
python train/train_learn_cmyw_to_rgb.py --split 'train' --root ./data/canon
python train/train_learn_cmyw_to_rgb.py \
    --root ./data/canon \
    --snapshot ./snapshots \
    --max_epochs 100 \
    --batch_size 24 \
    --cfa_size 4 \
    --mlr 1e-3 \
    --slr 2e-4 \
    --w_zhi 0.0001 \
    --alpha 10 \
    --bool_noise

Train on Real CMYW Dataset
python train/train_learn_cmyw_to_rgb_real.py --split 'train' --root ./data/real_cmyw

### Testing
Test on Synthetic Data
python Test/Test_learn_cmyw_to_rgb.py \
    --test_epoch lcmyw_20 \
    --split 'test' \
    --root ./data/canon \
    --save_image_dir ./results


Test on Real CMYW Dataset

python Test/Test_learn_cmyw_to_rgb_real.py \
    --test_epoch lcmyw_20 \
    --split 'test' \
    --root ./data/real_cmyw \
    --save_image_dir ./results

Key Parameters
Parameter	Description
--root	Dataset root directory
--snapshot	Model checkpoint saving directory
--save_image_dir	Reconstructed image saving directory
--test_epoch	Checkpoint file name for validation/testing
--continue_epoch	Checkpoint starting point for resuming training
--resume	Whether to resume training from checkpoint
--block_size	Image crop size for training (e.g., 128×128)
--cfa_size	CFA pattern size (e.g., 4 for 4×4)
--max_epochs	Total training epochs
--batch_size	Batch size
--mlr	Decoder network learning rate
--slr	Encoder (CFA) learning rate
--w_zhi	CFA weight initialization value
--alpha	Temperature coefficient for Gumbel-Softmax
--tau_g	Noise term temperature coefficient
--bool_noise	Whether to add noise during training
--noise_std	Noise standard deviation level
--loss	Loss function selection (L2 or ΔE)
--decoder	Decoder network selection (JDENet)
Evaluation Metrics
We use three full-reference metrics for quantitative evaluation:

PSNR (Peak Signal-to-Noise Ratio) — measures global quality attributes such as brightness and contrast

SSIM (Structural Similarity Index) — evaluates similarity by jointly considering luminance, contrast, and structural information

LPIPS (Learned Perceptual Image Patch Similarity) — captures subtle perceptual differences aligned with human visual perception

To evaluate your trained model:

python psnr_ssim/compute_metrics.py --pred_dir ./results --gt_dir ./data/test


