# prepare guidance model train_tuple.pkl : (data, label) 

python classifier_train.py --num_classes 1 --rnn_type gru --hidden_dim 256 --train_data data/mimic_icustay/train_tuple.pkl --val_data data/mimic_icustay/val_tuple.pkl



