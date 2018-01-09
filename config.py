import os


data_path = '/home/ubuntu/efs/CXR8'
all_img_path = os.path.join(data_path, 'images')
train_path = os.path.join(data_path, 'images_train')
test_path = os.path.join(data_path, 'images_test')
val_proportion = 10

use_gpu = True
train_batch_size = 8
val_batch_size = 16
workers = 4
