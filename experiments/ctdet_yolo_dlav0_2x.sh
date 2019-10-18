cd src
# train
python main.py ctdet --exp_id yolo_dlav0_2x --arch dlav0_34 --batch_size 48 --lr 1.875e-4 --gpus 4,5,6 --num_workers 16 --num_epochs 230 --lr_step 180,210 --dataset yolo --data_dir /home/amax/qszn/ciwa_train_data --scale 0.2 --rotate 180 --flip 0.5 --vflip 0.5
# test
python test.py ctdet --exp_id yolo_dlav0_2x --arch dlav0_34 --gpus 4 --dataset yolo --data_dir /home/amax/qszn/ciwa_train_data --keep_res --resume