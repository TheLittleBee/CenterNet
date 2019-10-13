cd src
# train
python main.py ctdet --exp_id yolo_dla_2x --batch_size 128 --lr 5e-4 --gpus 4,5,6 --num_workers 16 --num_epochs 230 --lr_step 180,210 --dataset yolo --data_dir /home/amax/qszn/ciwa_train_data --scale 0.2 --rotate 180 --flip 0.5 --vflip 0.5
