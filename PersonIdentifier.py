import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from data.base_dataset import BaseDataset, get_transform

class PersonIdentifier:
    """
    Get the person's masked image from the image.
    """
    def __init__(self):
        self.opt = TestOptions().parse()
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.model = "test"
        self.opt.loadSize = 256
        self.opt.output_nc = 20
        self.opt.which_epoch = 30
        self.opt.gpu_ids = 0
        self.model = create_model(self.opt)
        self.transform = get_transform(self.opt)
        if self.opt.which_direction == 'BtoA':
            self.input_nc = self.opt.output_nc
        else:
            self.input_nc = self.opt.input_nc

    def identify_person(self, image):
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        input = {'A': A, 'A_paths': A_path}
        self.model.set_input(input)
        self.model.test()
        fake_B = util.ndim_tensor2im(self.model.fake_B.data)
        return fake_B
