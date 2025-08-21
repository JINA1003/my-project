import os

for dir in ['service','data','submissions']:
    if not os.path.exists(dir) : 
        os.makedirs(dir)