# 5.3 Target-Aware Generation for Specific Downstream Tasks

This advanced mode enables **target-aware generation**, where the model produces time-series data that is **optimized to improve performance on a specific downstream task** (e.g., classification, detection). It integrates **gradient-based guidance** from a pre-trained classifier into the generation process, steering synthetic data toward task-relevant attributes.

This setup is useful when:
- You want synthetic data to enhance downstream task models
- You need to generate hard or rare samples for classifier robustness
- Controllability is required based on task-specific feedback

**Example Command:**

```bash
python inference.py \
  --base config.yaml \
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_guidance \
  --uncond \
  --downstream_pth_path ./classifier/checkpoints/best_model.pt \
  --guidance_path ./classifier/data/guidance_tuple.pkl
```

> `--use_guidance` enables classifier-informed generation  
> `--guidance_path` must point to a `.pkl` containing the guidance tuples  
> This assumes the classifier is already trained and stored at `downstream_pth_path`  

[🔙 Back to Main README](https://github.com/microsoft/TimeCraft)
