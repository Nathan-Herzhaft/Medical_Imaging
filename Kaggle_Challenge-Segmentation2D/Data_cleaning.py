# %%
import os

for item in os.listdir('data/train'):
    if ('mask' not in item) and ('_scan' not in item):
        file = os.path.join('data','train',item)
        new_name = item.replace('.tif','_scan.tif')
        new_file = os.path.join('data','train',new_name)
        os.rename(file,new_file)

# %%
