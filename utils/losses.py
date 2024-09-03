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
    
class FwdBwdLoss_simple(nn.Module):
    def __init__(self, B, F):
        super(FwdBwdLoss_simple, self).__init__()
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

class FwdBwdLoss(nn.Module):
    def __init__(self, B, F, k = 0, beta = 1):
        super(FwdBwdLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logsotmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)
        self.k = torch.tensor(k, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)

    def forward(self, inputs, z):
        v = inputs - torch.mean(inputs, axis = 1, keepdims = True)
        p = self.softmax(v)
        z = z.long()

        # Loss is computed as z'B'*phi(Ff)
        Ff = self.F @ p.T 
        log_Ff = torch.log(Ff+1e-8)
        B_log_Ff = self.B.T @ log_Ff
        L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))]) + 0.5 * self.k * torch.sum(torch.abs(v)**self.beta)
        #L = - torch.sum(B_log_Ff[z,range(B_log_Ff.size(1))]+1e-10)
        return L

class LBLoss(nn.Module):
    def __init__(self, k=1, beta=1):
        super(LBLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.k = k
        self.beta = beta
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, inputs, targets):
        device = inputs.device
        targets = targets.to(device)
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        logp = self.logsoftmax(v)
        L = - torch.sum(targets * logp) + 0.5 * self.k * torch.sum(torch.abs(v) ** self.beta)
        return L
    
class LBLoss_gpt4o(nn.Module):
    def __init__(self, k=1, beta=1):
        super(LBLoss_gpt4o, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.k = k
        self.beta = beta

    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        device = inputs.device
        targets = targets.to(device)

        # Centering the inputs
        v = inputs - torch.mean(inputs, axis=1, keepdims=True)
        
        # Compute log-softmax
        logp = self.logsoftmax(v)
        
        # Compute the loss
        L = - torch.sum(targets * logp) + 0.5 * self.k * torch.sum(torch.abs(v) ** self.beta)
        
        # Normalize by the batch size
        batch_size = inputs.size(0)
        L = L / batch_size
        
        return L

class EMLoss(nn.Module):
    def __init__(self,M):
        super(EMLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.M = torch.tensor(M)
        
    def forward(self,out,z):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logp = self.logsoftmax(out)

        p = torch.exp(logp)
        M_on_device = self.M.to(out.device)
        Q = p.detach() * M_on_device[z]
        #Q = p.detach() * torch.tensor(self.M[z])
        Q /= torch.sum(Q,dim=1,keepdim=True)

        L = -torch.sum(Q*logp)

        return L
    
class OSLCELoss(nn.Module):
    def __init__(self):
        super(OSLCELoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)

    def hardmax(self, A):
        #D = torch.eq(A, torch.max(A, axis=1, keepdims=True)[0])
        D = torch.eq(A, torch.max(A, axis=1, keepdims=True).values)
        return D.float() / torch.sum(D, axis=1, keepdims=True)

    def forward(self, inputs, targets):
        logp = self.logsoftmax(inputs)
        p = torch.exp(logp)
        num_classes = inputs.size(1)

        if targets.ndim == 1 or targets.size(1) == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()

        D = self.hardmax(targets * p)
        L = - torch.sum(D*logp)/ inputs.size(0)# Normalize by batch size
        return L