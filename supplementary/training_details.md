## Train TimeGen Framework

Use `main.py` for model training, `inference.py` for model inference and `visualize.py` for domain prompt visualization. 

The detailed descriptions about command line arguments are as follows:
| Parameter Name                    | Description                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `base` (`-b`)                     | Paths to base configuration files.                                                                                 |
| `train` (`-t`)                    | Boolean flag to enable training. (default: true)                                                                   |
| `debug` (`-d`)                    | Boolean flag to enter debug mode. (default: false)                                                                 |
| `seed` (`-s`)                     | Seed for initializing random number generators. (default: 23)                                                      |
| `logdir` (`-l`)                   | Directory for logging data. (default: ./logs)                                                                      |
| `seq_len` (`-sl`)                 | Sequence length for the model. (default: 24)                                                                       |
| `uncond` (`-uc`)                  | Boolean flag for unconditional generation.                                                                         |
| `use_pam` (`-up`)                 | Boolean flag to use the prototype assignment module.                                                               |
| `batch_size` (`-bs`)              | Batch size for training. (default: 128)                                                                            |
| `num_latents` (`-nl`)             | Number of latent variables. (default: 16)                                                                          |
| `overwrite_learning_rate` (`-lr`) | Learning rate to overwrite the config file. (default: None)                                                        |
| `gpus`                            | Comma-separated list of GPU ids to use for training.                                                               |
| `ckpt_name`                       | Checkpoint name to resume from for test or visualization. (default: last)                                          |
| `use_text`                       | Use text as condition                                          |


### Training and inference together
We provide end-to-end scripts that can be used for both training and inference.

```bash
python train_inference.py \
  --base config.yaml \
  --gpus 0, \
  --logdir ./logs/Your_Logidr  \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001  \
```

### Training with Prototypes and Text

```bash
python main.py \
  --base config.yaml \
  --gpus 0, \
  --logdir ./logs/Your_Logidr  \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001  \
  --use_text \
```

### Training with Prototypes
```bash
python main.py \
  --base config.yaml \
  --gpus 0,  \
  --logdir ./logs/Your_Logidr  \
  -sl 168  \
  -up  \
  -nl 16  \
  --batch_size 128 \
  -lr 0.0001  \
```

### Training with neither Prototypes nor Text

```bash
python main.py \
  --base config.yaml \
  --gpus 0,  \
  --logdir ./logs/Your_Logidr  \
  -sl 168  \
  -nl 16  \
  --batch_size 128 \
  -lr 0.0001  \
```
