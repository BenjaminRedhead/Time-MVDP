# TarDiff: Target-Oriented Diffusion Guidance for Synthetic Electronic Health Record Time Series Generation

  

## Introduction
![TarDiff OverView](./images/overview.png)

Synthetic Electronic Health Record (EHR) time-series generation is crucial for advancing clinical machine learning models, as it helps address data scarcity by providing more training data. However, most existing approaches focus primarily on replicating statistical distributions and temporal dependencies of real-world data. We argue that fidelity to observed data alone does not guarantee better model performance, as common patterns may dominate, limiting the representation of rare but important conditions. This highlights the need for generate synthetic samples to improve performance of specific clinical models to fulfill their target outcomes. To address this, we propose TarDiff, a novel target-oriented diffusion framework that integrates task-specific influence guidance into the synthetic data generation process. Unlike conventional approaches that mimic training data distributions, TarDiff optimizes synthetic samples by quantifying their expected contribution to improving downstream model performance through influence functions. Specifically, we measure the reduction in task-specific loss induced by synthetic samples and embed this influence gradient into the reverse diffusion process, thereby steering the generation towards utility-optimized data. Evaluated on six publicly available EHR datasets, TarDiff achieves state-of-the-art performance, outperforming existing methods by up to 20.4% in AUPRC and 18.4% in AUROC. Our results demonstrate that TarDiff not only preserves temporal fidelity but also enhances downstream model performance, offering a robust solution to data scarcity and class imbalance in healthcare analytics.


## 1 · Environment

Prepare TarDiff's environment.
```
conda env create -f environment.yml
conda activate tardiff
```

Prepare TS downstream task environment depands on the repo you used for the specific task.

## 2 · Data Pre-processing

You can access the raw datasets at the following links:

- [eICU Collaborative Research Database](https://eicu-crd.mit.edu/)
- [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)

> **Note:** Both datasets require prior approval and credentialing before download.

We focus exclusively on the multivariate time-series recordings available in these datasets.  
To assist with preprocessing, we provide high-level extraction scripts under **`data_preprocess/`**.



## 3 · Stage 1 · Train the *Base* Diffusion Model

```bash
bash train.sh            # trains TarDiff on MIMIC-III ICU-stay data
```

> **Edit tip:** open the example YAML in `configs/base/` and replace any placeholder data paths with your own before running.

This step produces an unconditional diffusion model checkpoint—no guidance yet.

---

## 4 · Stage 2 · Train a Downstream Task Model (Guidance Source)

An example RNN classifier is supplied in **`classifier/`**.

```bash
cd classifier
bash train.sh            # saves weights to classifier/checkpoint/
cd ..
```

Feel free to swap in any architecture that suits your task.

---

## 5 · Stage 3 · Target-Guided Generation

With **both** checkpoints ready—the diffusion backbone and the task model—start guided sampling:

```bash
bash generation.sh       # remember to update paths to both weights
```

The script creates a synthetic dataset tailored to the guidance task.

---

## 6 · Stage 4 · Utility Evaluation — *TSTR* and *TSRTR*

After generation, you can assess the utility of the synthetic data for the **target task** using two complementary protocols:

| Protocol | Training Set | Test Set | Question Answered |
|:---------|:-------------|:---------|:------------------|
| **TSTR** (Train-Synthetic, Test-Real) | **Synthetic only** | **Real** | “If I train a model purely on synthetic EHRs, how well does it generalize to real patients?” |
| **TSRTR** (Train-Synthetic-Real, Test-Real) | **Synthetic + Real** (α ∈ {0.2, …, 1.0}) | **Real** | “If I augment the real training set with α× synthetic samples, does it improve model performance?” |

---

### How to run TSTR and TSRTR evaluations

You can directly reuse the training script under `classifier/` to run both evaluations:

```bash
cd classifier
bash train.sh    # Edit the training data path to point to either synthetic-only or mixed (real + synthetic) data
```

- For **TSTR**, set the training set path to the synthetic dataset.
- For **TSRTR**, combine the real training data with synthetic samples according to your desired α ratio, and update the path accordingly.

The downstream model will be trained and evaluated automatically on the real validation and test sets.

---

Enjoy exploring target-oriented diffusion for healthcare ML!  For issues or pull requests, please open a GitHub ticket.
