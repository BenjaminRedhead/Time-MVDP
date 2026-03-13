# 5.1 Controllable Generation with Domain Prompts

In this mode, TimeCraft leverages learned **semantic prototypes** (also referred to as "domain prompts") to control the generation of synthetic time-series data. These prototypes encode structural or categorical properties of specific domains (e.g., medical, financial, climate), enabling the model to generate data that conforms to domain-specific characteristics without relying on explicit textual input.

**Example Command:**

```bash
python inference.py \
  --base config.yaml \
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_pam
```

> `--use_pam` enables prototype-based control  

Inference with unconditional

**Example Command:**

```bash
python inference.py \
  --base config.yaml \ 
  --resume true \
  --ckpt_name ./checkpoints/ \
```

[🔙 Back to Main README](https://github.com/microsoft/TimeCraft)
