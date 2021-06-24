# Reference: https://github.com/YerevaNN/mimic3-benchmarks
# In-hospital mortality prediction
## Trian: 
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality
## Inference:
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality
## Send model folder from cloud to edge
# Decompensation prediction
## Train:
python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode train --batch_size 8 --output_dir mimic3models/decompensation
## Inference:
python -um mimic3models.decompensation.main --network mimic3models/keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode test --batch_size 8 --output_dir mimic3models/decompensation
## Send model folder from cloud to edge
# Phenotype classification
## Train:
python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/phenotyping
## Inference:
python -um mimic3models.phenotyping.main --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/phenotyping
## Send model folder from cloud to edge
