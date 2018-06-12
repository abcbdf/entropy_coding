import torch
import h5py
from torch import nn
from scipy.misc import imresize
import torch
import cv2
from PIL import Image
import os
import logging
from multiprocessing import Pool
import numpy as np
import time
import random
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file, phase):
        self.iter = 4
        self.file = get_image_list(file, phase)
        pool = Pool()
        self.datas = pool.map(self.getitem, range(len(self.file)))
        pool.close()
        pool.join()
        self.phase = phase
        
    def __getitem__(self, id):
        #print(self.datas[id].shape)
        return self.datas[int(id / 4)][id % 4, 0, :, :, :]
        # real_id = int(id / 256)
        # logging.debug(self.file[real_id])
        # temp = self.datas[real_id]
        # x = int((id - real_id * 256) / 16)
        # y = id - real_id * 256 - x * 16
        # return temp[:, x : x + 32, y : y + 32]
    
    def __len__(self):
        return len(self.file) * self.iter
        

    def getitem(self, id):
        print(self.file[id])   
        content = np.load(self.file[id])
        codes = np.unpackbits(content['codes'])
        codes = np.reshape(codes, content['shape']).astype(np.float32)
        return codes

def get_image_list(train_dir, phase):
    image_list = []
    index = 0
    for dir in os.listdir(train_dir):
    	if(phase == 'train'):
    		index += 1
    		image_list.append(os.path.abspath(train_dir + dir))
    		if(index > 1000):
        		break
    	elif(phase == 'val'):
    		index += 1
    		image_list.append(os.path.abspath(train_dir + dir))
    		if(index > 10):
    			break
    return image_list
        
