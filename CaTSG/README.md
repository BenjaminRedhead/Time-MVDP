# CaTSG for Causal Time Series Generation

This repository provides the implementation code corresponding to our paper entitled [Causal Time Series Generation via Diffusion Models](https://arxiv.org/pdf/2509.20846).
The code is implemented using PyTorch 1.13.0 and PyTorch Lightning 1.4.2 framework on a server with NVIDIA A100 80GB PCIe.

## Description
In the paper, we introduce *causal time series generation* as a new time series generation task family, formalized within Pearl’s causal ladder, include *interventional* and *counterfactual* settings. 

To instantiate these tasks, we develop **CaTSG**, a unified diffusion-based generative framework with backdoor-adjusted guidance that steers sampling toward desired interventional and individual counterfactual distributions. 

![image](./assets/framework.png)

## Installation

### Requirements
CaTSG uses the following dependencies:
- Pytorch 1.13.0 and PyTorch Lightning 1.4.2
- Numpy and Scipy
- Python 3.8
- CUDA 11.7 or latest version, cuDNN

### Setup Environment

Please first clone the **TimeCraft** repository and then set up the environment for CaTSG.

```bash
# Clone the repository
git clone https://github.com/microsoft/TimeCraft.git
cd TimeCraft/CaTSG

# Create and activate conda environment
conda env create -f environment.yml
conda activate catsg
```

## Dataset Preparation

This project supports both **synthetic datasets** for controlled experiments and **real-world datasets** for practical evaluation.  

### Overview

- **Synthetic datasets**  
We construct two synthetic datasets which simulate a class of damped mechanical oscillators governed by second-order differential equations $m \cdot \ddot{x}(t) + \gamma \cdot \dot{x}(t) + k \cdot x(t) = 0$. Details are presented in the appendix of our paper.
  - **Harmonic-VM**: Harmonic Oscillator with Variable Mass
  - **Harmonic-VP**: Harmonic Oscillator with Variable Parameters  

- **Real-world datasets**  
  - **[Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)**: 
  Four years of hourly air quality and meteorological measurements from **12 monitoring stations** in Beijing, China.
  - **[Traffic](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)**: 
  Hourly traffic volume recorded on Interstate 94 near Minneapolis–St Paul, USA, including weather and holiday indicators.  

### Synthetic Datasets

You can also create the datasets from scratch:  
```bash
# Harmonic-VM
python utils/tsg_dataset_creator.py --config configs/dataset_config/harmonic_vm.yaml

# Harmonic-VP
python utils/tsg_dataset_creator.py --config configs/dataset_config/harmonic_vp.yaml
```

### Real-World Datasets

#### Step 1: Download Raw Data

- **Air Quality**: Download [here](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) and unzip the dataset. Place all `.csv` files from `PRSA_Data_20130301-20170228`(12 statations data) into `./data_raw/AQ/` folder.
- **Traffic**: Download [here](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and unzip the dataset. Place the single csv file into `./data_raw/Metro_Interstate_Traffic_Volume.csv`.

After downloading, the directory should look like:

  ```bash
  data_raw
  ├── AQ
  │   ├── PRSA_Data_Aotizhongxin_20130301-20170228.csv
  │   ├── ...
  │   └── PRSA_Data_Wanshouxigong_20130301-20170228.csv
  └── Metro_Interstate_Traffic_Volume.csv
  ```

#### Step 2: Preprocess into Required Format
Run the following commands to generate processed datasets:
```bash
# Air Quality dataset
python utils/tsg_dataset_creator.py --config configs/dataset_config/aq.yaml

# Traffic dataset
python utils/tsg_dataset_creator.py --config configs/dataset_config/traffic.yaml
```

### Dataset split

The default dataset splits used in our experiments are listed below.
You can modify them in `configs/dataset_config/{dataset}.yaml`.
For the Air Quality dataset split, an interactive map of station locations is available [**here**](./assets/aq_station_loc.html).

| Type       | Dataset     | Target ($x$)   | Context ($c$)                                         | Default split strategy                                                                                                                                                                                                                                                                            | Samples (Train/Val/Test) |
|------------|-------------|----------------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
|  Synthetic | Harmonic-VM | Acceleration   | Velocity, Position                                    | $\alpha$-based:      Train $[0.0,0.2]$;      Val $[0.3, 0.5]$;      Test $[0.6,1.0]$                                                                                                                                                                                                         | 3,000/ 1,000/ 1,000      |
|  Synthetic | Harmonic-VP | Acceleration   | Velocity, Position                                    | Combination-based:      Train: $\alpha \in [0.0, 0.2]$, $\beta \in [0.0, 0.01]$, $\eta \in [0.002,   0.08]$      Val:  $\alpha \in [0.3, 0.5]$, $\beta   \in [0.018, 0.022]$, $\eta \in [0.18, 0.22]$      Test: $\alpha \in [0.6, 1.0]$, $\beta \in [0.035, 0.04]$, $\eta \in [0.42,   0.5]$ | 3,000/ 1,000/ 1,000      |
| Real-world | Air Quality | $PM_{2.5}$     | TEMP, PRES, DEWP, WSPM, RAIN, wd                      | Station-based: Train (Dongsi,   Guanyuan, Tiantan, Wanshouxigong, Aotizhongxin, Nongzhanguan, Wanliu,   Gucheng); Val (Changping, Dingling); Test (Shunyi, Huairou)                                                                                                                          | 11,664/2,916/2,916          |
| Real-world | Traffic     | traffic_volume | rain_1h, snow_1h, clouds_all,   weather_main, holiday | Temperature-based: Train   (<12°C); Val ([12,22]°C); Test (>22°C)                                                                                                                                                                                                                            | 26,477/16,054/5,578         |

## Quick Start

Train CaTSG on the harmonic dataset and test both intervention and counterfactual tasks:

```bash
# 1) Training (automatically runs both int and cf evaluation after training)
python main.py --base configs/catsg.yaml --dataset harmonic_vm --train

# 2) Testing specific tasks
python main.py --base configs/catsg.yaml --dataset harmonic_vm --test int
python main.py --base configs/catsg.yaml --dataset harmonic_vm --test cf_harmonic
```

**Outputs**
- **Logs**: saved under `logs/<dataset>/CaTSG/<exp_name>/` 
- **Results**: saved as `.csv` under `results/<dataset>/`

## Citation

If you find our work useful, please cite:

```bibtex
@article{xia2025causal,
  title={Causal Time Series Generation via Diffusion Models},
  author={Xia, Yutong and Xu, Chang and Liang, Yuxuan and Wen, Qingsong and Zimmermann, Roger and Bian, Jiang},
  journal={arXiv preprint arXiv:2509.20846},
  year={2025}
}
```
