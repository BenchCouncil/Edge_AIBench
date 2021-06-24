#/bin/bash
# execute for different datasize on edge 
a=(4 8 16 32 64 128 256 512 1024 2048)
for i in ${a[@]}
do
    for((j=0;j<1;j++))
    do
	#batch_size not changed
    python3 -um mimic3models.phenotyping.inference --network mimic3models/keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --datasize $i --load_state mimic3models/phenotyping/keras_states/k_lstm.n256.d0.3.dep1.bs8.ts1.0.epoch21.test0.34822362972669896.state --batch_size 8 --output_dir mimic3models/phenotyping/ >> experiments/EdgeVsServer/phenotyping_edge.txt
    #batch_size changed with datasize
    #python3 -um mimic3models.in_hospital_mortality.inference --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --datasize $i --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size $i --output_dir mimic3models/in_hospital_mortality/ >> experiments/EdgeVsServer/in_hospital_mortality_edge_batchsize.txt
    #datasize not changed,batchsize changed
    #python3 -um mimic3models.decompensation.inference --network mimic3models/keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --dropout 0.3 --mode test --datasize 2048 --load_state mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state --batch_size $i --output_dir mimic3models/in_hospital_mortality/ >> experiments/EdgeVsServer/in_hospital_mortality_edge_onlybatchsize.txt
    done
#    sudo sysctl -w vm.drop_caches=3
done

