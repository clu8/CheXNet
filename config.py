import os


data_path = '/home/ubuntu/efs/CXR8'
all_img_path = os.path.join(data_path, 'images')
train_path = os.path.join(data_path, 'images_train')
test_path = os.path.join(data_path, 'images_test')
val_proportion = 8 # 70% train, 10% val

model_path = 'model.pkl'

use_gpu = True
workers = 4

num_epochs = 40
train_batch_size = 16
val_batch_size = 16

