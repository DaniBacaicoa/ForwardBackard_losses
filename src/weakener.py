"""" Transforms a dataset with the true labels into a weakly labeled dataset
    The weakening process is given by a Mixing matrix, a.k.a., Transition matrix
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
import numpy as np
import torch
import cvxpy

from collections import Counter

class Weakener(object):
    def __init__(self, true_classes):

        # Dimentions of the problem
        self.c = true_classes
        self.d = None
        self.h = None

        # Matrices
        self.M = None
            #For FB losses
        self.B = None
        self.B_opt = None
        self.F = None
        
            #For Backward losses
        self.Y = None
        self.Y_opt = None
        self.Y_conv = None
        self.Y_opt_conv = None

    def generate_M(self, model_class = 'pll', alpha = 1, beta = None, corr_p = 0.5, corr_n = None):
        self.corr_p = corr_p
        self.corr_n = corr_n

        if model_class == 'Noisy_Patrini_MNIST':
            # Nose is: 2 -> 7; 3 -> 8; 5 <-> 6; 7 -> 1
            self.M = torch.tensor([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , self.corr_p, 0. , 0. ],
                      [0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 1-self.corr_p, self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , self.corr_p, 1-self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1-self.corr_p, 0. , 0. ],
                      [0. , 0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
        elif model_class == 'Noisy_Patrini_CIFAR10':
            #TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → HORSE, CAT ↔ DOG.
            self.M = torch.tensor([[1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , self.corr_p ],
                      [0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1-self.corr_p , 0. , self.corr_p , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , self.corr_p , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , self.corr_p , 0. , 0. , 1. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1-self.corr_p ]])
        elif model_class == 'Noisy_Natarajan':
            self.M = torch.tensor([
                [1-self.corr_n, self.corr_p  ],
                [self.corr_n  , 1-self.corr_p]])
        elif


