#/bin/bash
"""
for((i=0;i<1;i++))
do
    python -um mimic3models.in_hospital_mortality.inference --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --datasize small --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size 32 --output_dir mimic3models/in_hospital_mortality >> experiments/EdgeVsServer/in_hospital_mortality_test_result.txt
done
for((i=0;i<1;i++))
do
    python -um mimic3models.in_hospital_mortality.inference --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --datasize large --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size 32 --output_dir mimic3models/in_hospital_mortality >> experiments/EdgeVsServer/in_hospital_mortality_test_result.txt
done
"""
for((i=0;i<10;i++))
do
    python -um mimic3models.in_hospital_mortality.inference --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --datasize small --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size 32 --output_dir mimic3models/in_hospital_mortality >> experiments/EdgeVsServer/in_hospital_mortality_small_result.txt
done
for((i=0;i<10;i++))
do
    python -um mimic3models.in_hospital_mortality.inference --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --datasize large --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size 32 --output_dir mimic3models/in_hospital_mortality >> experiments/EdgeVsServer/in_hospital_mortality_large_result.txt
done
