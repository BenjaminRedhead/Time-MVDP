# prepare guidance model train_tuple.pkl : (data, label) 

python classifier/classifier_train.py --num_classes 1 --rnn_type gru --hidden_dim 256 --train_data /data/train_tuple.pkl --val_data /data/val_tuple.pkl



