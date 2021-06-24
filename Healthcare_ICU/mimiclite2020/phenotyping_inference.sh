
python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --batch_size 8 --load_state mimic3models/phenotyping/keras_states/k_lstm.n256.d0.3.dep1.bs8.ts1.0.epoch23.test0.3491972863225094.state --output_dir mimic3models/phenotyping
