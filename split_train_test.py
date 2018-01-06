import os
import shutil

import config


with open(os.path.join(config.data_path, 'train_val_list.txt'), 'r') as f:
    for img_file in f:
        img_file = img_file.rstrip()
        shutil.copy2(
            os.path.join(config.all_img_path, img_file),
            os.path.join(config.train_path, img_file)
        )

with open(os.path.join(config.data_path, 'test_list.txt'), 'r') as f:
    for img_file in f:
        img_file = img_file.rstrip()
        shutil.copy2(
            os.path.join(config.all_img_path, img_file),
            os.path.join(config.test_path, img_file)
        )
