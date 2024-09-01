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
        if corr_n == None:
            self.corr_n = corr_p
        else:
            self.corr_n = corr_n

        if model_class == 'Noisy_Patrini_MNIST':
            # Nose is: 2 -> 7; 3 -> 8; 5 <-> 6; 7 -> 1
            #self.M = torch.tensor([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
            self.M = np.array([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
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
            #self.M = torch.tensor([[1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
            self.M = np.array([[1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
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
            #self.M = torch.tensor([
            self.M = np.array([
                [1-self.corr_n, self.corr_p  ],
                [self.corr_n  , 1-self.corr_p]])
        elif model_class == 'pu':
            if self.c > 2:
                raise NameError('PU corruption coud only be applied when tne number o true classes is 2')
                # [TBD] if alpha is a vector raise error
                alpha = [alpha, 0]
            M = np.eye(2) + alpha * np.ones((2, 2))
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M
        # c = d
        elif model_class == 'supervised':
            M = np.identity(self.c)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'noisy':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more prbable. 
                As a limiting case supervised is achieved as alpha -> infty
            [  1+a_1  a_2    a_3  ]
            [  a_1    1+a_2  a_3  ]
            [  a_1    a_2    1+a_3]
            '''
            if any(np.array(alpha) < -1):
                NameError('For noisy labels all components of alpha should be greater than -1')
            elif any(np.array(alpha) == -1):
                cl = np.where(np.array(alpha) == -1)[0]
                print('labels', cl, 'are considered complemetary labels')
                # warning('Some (or all) of the components is considered as complemetary labels')
            M = np.eye(self.c) + alpha * np.ones(self.c)
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M

        elif model_class == 'complementary':
            '''
            This gives one of de non correct label 
            '''
            M = (1 - np.eye(c)) / (c - 1)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M

        # c < d
        elif model_class == 'weak':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more probable. 
                As a limiting case supervised is achieved as alpha -> infty

             z\y  001    010    100
            000[  a_1    a_2    a_3  ]
            001[  1+a_1  a_2    a_3  ]
            010[  a_1    1+a_2  a_3  ]
            001[  a_1    a_2    a_3  ]
            011[  a_1    a_2    a_3  ]
            100[  a_1    a_2    1+a_3]
            101[  a_1    a_2    a_3  ]
            111[  a_1    a_2    a_3  ]
            '''
            M = np.zeros((2 ** self.c, self.c))
            for i in range(self.c):
                M[2 ** i, i] = 1
            M = alpha * M + np.ones((2 ** self.c, self.c))
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M


        elif model_class == 'pll':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021)
            probs, Z = self.pll_weights(p=self.pll_p)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            M = M / M.sum(0)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'pll_a':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021) they don't allow anchor points but this method does.
            probs, Z = self.pll_weights(p=self.pll_p, anchor_points=True)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            M = M / M.sum(0)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'Complementary_weak':
            '''
            This gives a set of candidate labels over the non correct one.
            '''
            # [TBD]
            return _

        self.M, self.Z, self.labels = self.label_matrix(M)
        self.d = self.M.shape[0]
        self.Y = np.linalg.pinv(self.M)
        self.Y_opt = self.virtual_matrix(self, p=None, optimize = True, convex=False)
        self.Y_conv = self.virtual_matrix(self, p=None, optimize = False, convex=True)
        self.Y_opt_conv = self.virtual_matrix(self, p=None, optimize = True, convex=True)

    def generate_weak(self, y, seed=None):
        # It should work with torch
        # the version of np.random.choice changed in 1.7.0 that could raise an error-
        d, c = self.M.shape
        # [TBD] include seed
        self.z = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in torch.max(y, axis=1)[1]]).to(torch.int32)

        self.w = torch.from_numpy(self.Z[self.z.to(torch.int32)] + 0.)
        return self.z, self.w


    def virtual_matrix(self, p=None, optimize = True, convex=True):
        d, c = self.M.shape
        I_c = np.eye(c)

        if p == None:
            if optimize:
                p = self.generate_wl_priors(self.z)
            else:
                p = np.ones(d)/d
        c_1 = np.ones((c,1))
        d_1 = np.ones((d,1))

        hat_Y = cvxpy.Variable((c,d))

        if c==d:
            self.Y = np.linalg.pinv(self.M)
        elif convex:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c, hat_Y.T @ c_1 == d_1]
            )
            prob.solve(solver=cvxpy.CLARABEL)
            Y = hat_Y.value
        else:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c]
            )
            prob.solve(solver=cvxpy.CLARABEL)
            Y = hat_Y.value
        
        return Y



    def virtual_labels(self, y = None, p=None, optimize = True, convex=True):
        '''
        z must be the weak label in the z form given by generate weak
        '''
        #In order to not generate weak labels each time we seek the existence of them
        # and in the case they are already generated we don't generate them again
        if self.z is None:
            if y is None:
                raise NameError('The weak labels have not been yet created. You shuold give the true labels. Try:\n  class.virtual_labels(y)\n instead')
            _,_ = self.generate_weak(y)
        if self.Y is None:
            self.virtual_matrix(p, optimize, convex)
        self.v = self.Y.T[self.z]
        return
    def generate_wl_priors(self, loss = 'CELoss'):

        #z_count = Counter(z)
        #p_est = np.array([z_count[x] for x in range(self.d)])
        p_est = np.array(self.z.bincount(minlength=self.Z.shape[0]))
        #p_est = np.array(torch.bincount(self.z))
        v_eta = cvxpy.Variable(int(self.c))
        if loss == 'CELoss':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        else:
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)

        problem = cvxpy.Problem(cvxpy.Minimize(lossf),
                                [v_eta >= 0, np.ones(self.c) @ v_eta == 1])
        problem.solve(solver=cvxpy.CLARABEL)

        # Compute the wl prior estimate
        p_reg = self.M @ v_eta.value

        return p_reg
    '''
        v_eta = cvxpy.Variable(self.c)
        if loss == 'cross_entropy':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        elif loss == 'square_error':
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)
        '''

    def label_matrix(self, M):
        """
        The objective of this function is twofold:
            1. It removes rows with no positive elements from M
            2. It creates a label matrix and a label dictionary

        Args:
            M (numpy.ndarray): A mixing matrix (Its not required an stochastic matrix).
                but its required its shape to be either dxc(all weak labels) or cxc(all true labels)

        Returns:
            - numpy.ndarray: Trimmed verison of the mixing matrix.
            - numpy.ndarray: Label matrix, where each row is converted to a binary label.
            - dict: A dictionary of labels where keys are indices and values are binary labels.

        Example:
            >>> M = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0],
                              [0, 1, 1], [1, 0, 1], [0, 1, 0],
                              [0, 1, 1], [0, 0, 0]])
            >>> trimmed_M, label_M, labels = label_matrix(M)
            >>> trimmed_M
            array([[1 0 0],
                   [0 1 1],
                   [1 0 1],
                   [0 1 0],
                   [0 1 1]])
            >>> label_M
            array([[0 0 0],
                   [0 1 1]
                   [1 0 0]
                   [1 0 1]
                   [1 1 0]])
            >>> labels
            {0: '000', 1: '011', 2: '100', 3: '101', 4: '110'}
        """
        d, c = M.shape

        if d == c:
            # If M is a square matrix, labels are
            Z = np.eye(c)
            # We make this in reversin order to get ('10..00', '01..00', .., '00..01')
            labels = {i: format(2**(c-(i+1)),'b').zfill(c) for i in range(c)} 
        elif (d<2**c):
            raise ValueError("Labels cannot be assigned to each row")
        else:
            # Z is a matrix with all the possible labels
            Z = np.array([[int(i) for i in format(j,'b').zfill(c)] for j in range(2**c)])
            # Now, we will get only the rows with nonzero elements
            z_row = M.any(axis = 1)
            # We assing the binary representation to those nonzero rows
            encoding = [format(i,'b').zfill(c) for i, exists in enumerate(z_row) if exists]
            # and we will give a numerical value to those representation of labels
            labels = {i:enc for i,enc in enumerate(encoding)}
            Z = Z[z_row,:]
            M = M[z_row,:]

        return M, Z, labels

    def pll_weights(self, c=None , p=0.5, anchor_points=False):
        """
        Descrip

        Args:
            p (double): 

        Returns:
            - dict: 
            - numpy.ndarray: 

        Example:
            >>> p = 
            >>> probs, Z= label_matrix(pll_weights)
            >>> probs
            output
            >>> z
            out
        """
        if c is None:
            c = self.c
        _, Z, _ = self.label_matrix(np.ones((2 ** c, c)))
        probs = {0: 0}
        q = 1 - p
        
        if anchor_points:
            probs[1] = q ** c + p * q ** (c - 1)
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1)
        else:
            probs[1] = 0
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1) + (q ** c + p * q ** (c - 1)) / (c - 1)
        for i in range(1, c + 1):
            probs[i] = p ** i * q ** (c - i) + p ** (i - 1) * q ** (c - i + 1)
        return probs, np.array(Z)
