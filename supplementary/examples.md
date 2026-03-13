
## Example Settings and Expected Results

### Demo training and inference of TimeGen with prototypes and text

```bash
python training_inference.py \
  --base text_control.yaml \
  --gpus 0, \
  --logdir ./logs/  \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001  \
  --use_text \
```

### Demo training and inference of TimeGen with prototypes

```bash
python training_inference.py \
  --base multi_domain_timedp.yaml  \
  --gpus 0,  \
  --logdir ./logs/ \
  -sl 168  \
  -up  \
  -nl 16  \
  --batch_size 128 \
  -lr 0.0001  \
```

### Demo training and inference of TimeGen with text

```bash
python train_inference.py \
  --base text_control.yaml\
  --gpus 0,  \
  --logdir ./logs/ \
  -sl 168  \
  -nl 16  \
  --batch_size 128 \
  -lr 0.0001  \
  -use_text \
```

### Demo training and inference of TimeGen without text and prototype

```bash
python train_inference.py \
  --base multi_domain_timedp.yaml\
  --gpus 0,  \
  --logdir ./logs/ \
  -sl 168  \
  -nl 16  \
  --batch_size 128 \
  -lr 0.0001  \
```

### Dataset

The [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) is a public multivariate time series dataset widely used for forecasting, anomaly detection, and energy consumption analysis. It contains 15-minute interval electricity consumption records (in kWh) from 370 industrial and residential clients of a Portuguese energy provider, collected between 2011 and 2014. The diverse consumption patterns make it ideal for evaluating machine learning models in multivariate time series forecasting and classification.
### Example Output

We evaluate **TimeGen** and baseline models on time series generation tasks. The metrics used are Maximum Mean Discrepancy (MDD) and Kullback-Leibler divergence (K-L), both measuring the similarity between the generated and real data distributions—lower values indicate better performance. **TimeGen**  consistently outperforms existing baselines across both metrics. Combining prototypes and text leads to the best results, showing the advantage of integrating structured temporal patterns with semantic information. 

| Model       | mdd | k-l |
|----------------|--------|-----------|
| TimeGen with prototypes and text  | 0.222  | 0.012   | 
| TimeGen with prototypes   | 0.237  | 0.016    |
| TimeGen with text   | 0.288    | 0.021     |
| TimeGAN   | 1.631   | 1.389       |
| GT-GAN   | 1.290    | 0.956       |
| TimeVAE   | 0.978   | 0.206       |

Note: The BRIDGE implementation of TimeGen uses a much smaller training dataset compared to TimeDP, due to trade-offs in handling large-scale textual data.

