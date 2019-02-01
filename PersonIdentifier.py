import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model 
from util.visualizer import Visualizer
from util import util 
from data.base_dataset import BaseDataset, get_transform
import cv2
from PIL import Image
import sys

class PersonIdentifier:
    """
    Get the person's masked image from the image.
    """
    def __init__(self):
        sys.argv = ["PersonIdentifier.py"]
        opt = TestOptions().parse()
        opt.dataroot = "../Human"
        opt.dataset = "LIP"
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.model = "test"
        opt.loadSize = 256
        opt.output_nc = 20
        opt.which_epoch = 30
        opt.checkpoints_dir = '/home/ubuntu/suraj/package/MMAN/checkpoints'
        opt.name = 'Exp_0'
        #opt.gpu_ids = '0'
        opt.dataset_mode = 'single'
        if opt.which_direction == 'BtoA':
            self.input_nc = opt.output_nc
        else:
            self.input_nc = opt.input_nc
        self.model = create_model(opt)
        self.transform = get_transform(opt)

    def filter_background(self, cv2_im):
        """
        This function will filter background of person's body.
        """
        pil_image = Image.fromarray(cv2_im)
        A_img = pil_image.convert('RGB')
        A = self.transform(A_img)
        if self.input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        A = A.unsqueeze(0)
        A_path = " "
        input = {'A': A, 'A_paths': A_path}
        #print(A)
        self.model.set_input(input)
        self.model.test()
        fake_B = util.ndim_tensor2im(self.model.fake_B.data)
        resize_img = cv2.resize(fake_B, (cv2_im.shape[1],cv2_im.shape[0]), interpolation = cv2.INTER_AREA)
        res  = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)        
        vid_bw = cv2.threshold(res, 43, 255, cv2.THRESH_BINARY_INV)[1]
        img2_fg = cv2.bitwise_or(cv2_im,(255,255,255) ,cv2_im, mask = vid_bw)
        return img2_fg
