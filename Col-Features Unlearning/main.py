#!/usr/bin/python3

"""main.py Contains an implementation of split learning using a
                     slighlty modified version of the LeNet5 CNN architecture.

                     Split learning is here implemented for multiple workers and
                     one central server, using the Message Passing Interface
                     (MPI). This implementation has been inspired by the
                     private repository of Abishek Shing as part of the work of
                     Vepakomma et al. on split learning, please see:

                     Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and
                        Ramesh Raskar. Split learning for health: Distributed
                        deep learning without sharing raw patient data.
                        arXiv preprint arXiv:1812.00564 , 2018

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Zhaobo Lu"

import numpy as np
import pandas as pd
import torch
import utils
from federated_learning import FederateLearning, FederatedUnlearningFeatures
from options import args_parser
import time
BATCH_SIZE = 1000


if __name__ == "__main__":
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.retain = True
    utils.set_random_seeds(args.seed)
    framework = FederateLearning(args)
    framework.federated_train()
    framework.finetuning()
    # framework.vertical_unlearning()
    # framework.retain()

