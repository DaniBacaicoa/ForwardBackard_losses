import torch
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp)
        return L
    
class FwdLoss(nn.Module):
    def __init__(self, M):
        super(FwdLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsotmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.M = torch.tensor(M, dtype=torch.float32, device=device)

    def forward(self, inputs, z):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        z = z.long()

        # Loss is computed as phi(Mf)
        Mp = self.M @ p.T
        L = - torch.sum(torch.log(Mp[z,range(Mp.size(1))]))
        #L = - torch.sum(torch.log(Mp[z,range(Mp.size(1))]+1e-10))
        return L
    
class FwdBwdLoss(nn.Module):
    def __init__(self, B, F):
        super(FwdLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsotmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)

    def forward(self, inputs, z):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        z = z.long()

        # Loss is computed as z'B'*phi(Ff)
        Ff = self.F @ p.T 
        log_Ff = torch.log(Ff)
        B_log_Ff = self.B.T @ log_Ff
        L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))])
        #L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))]+1e-10)
        return L

# Bwd = FwdBwdLoss(pinv(M),I_c)
# Fwd = FwdBwdLoss(I_d,M)