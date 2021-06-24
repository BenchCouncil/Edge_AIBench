# Reference:
https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial
# Prepare:
wget http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip
unzip Market-1501-v15.09.15.zip
edit the script prepare.py in the editor. Change the fifth line in prepare.py to your download path
python prepare.py
# Train:
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  Market-1501-v15.09.15/pytorch/
# Inference:
python test.py --name ft_ResNet50 --test_dir Market-1501-v15.09.15/pytorch/  --batchsize 32 --which_epoch 59
# Send model folder from cloud to edge 
