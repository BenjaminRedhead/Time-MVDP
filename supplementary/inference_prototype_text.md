# 5.2 Controllable Generation with Domain Prompts and Text

This mode combines **textual conditioning** with semantic **prototypes** to offer more expressive and fine-grained control over the generated time series. By including natural language prompts, users can specify high-level trends (e.g., "a rising curve with seasonal dips") or domain-specific features (e.g., "heartbeat pattern after exercise").

This setting is especially powerful when:
- Users want to guide generation using natural language descriptions
- Additional domain knowledge needs to be injected beyond prototypes

**Example Command:**

```bash
python inference.py \
  --base config.yaml \
  --resume true \
  --ckpt_name ./checkpoints/ \
  --use_pam \
  --use_text \
  --text_emb_dir ./your_text_embedding_dir/
```

> `--use_pam` + `--use_text` = joint prototype-text conditioning  
> `--text_emb_dir` points to pre-computed text embeddings  

[🔙 Back to Main README](https://github.com/microsoft/TimeCraft)
