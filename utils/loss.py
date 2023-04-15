import torch
import torch.nn as nn
import numpy as np
from .meshgen import meshgen

class Supervised_LossFunc(nn.Module):
    def __init__(self):
        super(Supervised_LossFunc, self).__init__()
        pass
        #self.weight = weight

    def forward(self, pred, target):
        hor1 = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        ver1 = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        ver1 = ver1.permute(0,1,3,2)
        diag1 = pred[:, :, 1:, :-1] - pred[:, :, :-1, 1:]
        cat1 = torch.cat((hor1, ver1, diag1), dim = 2)

        #print(dis_pred)

        hor2 = target[:, :, :, 1:] - target[:, :, :, :-1]
        ver2 = target[:, :, 1:, :] - target[:, :, :-1, :]
        ver2 = ver2.permute(0,1,3,2)
        diag2 = target[:, :, 1:, :-1] - target[:, :, :-1, 1:]
        cat2 = torch.cat((hor2, ver2, diag2), dim = 2)

        dis = cat1 - cat2
        # Do not use dis_loss = torch.mean(torch.sqrt((dis[:, 0])**2 + (dis[:, 1])**2)), the gradient of which will become nan when the norm of a vector is equal to 0.
        dis_loss = torch.mean(torch.norm(dis, dim = 1))
        return dis_loss

        #return F.mse_loss(dis_pred, dis_true, reduction='mean')

class BSLossFunc(nn.Module):
    def __init__(self):
        super(BSLossFunc, self).__init__()
        self.unfold = nn.Unfold((3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, pred, A):
        # Here batch_size is 2 * the real batch size
        N, _, h, w = pred.shape

        pred_unfold = self.unfold(pred).reshape((N, 2, 9, h, w))

        x = pred_unfold[:, 0]
        y = pred_unfold[:, 1]
        bx_hat = torch.sum(x[:, 1:-1, :, 1:-1] * A[:, :, :, 1:-1], dim=1)
        by_hat = torch.sum(y[:, 1:-1, 1:-1, :] * A[:, :, 1:-1, :], dim=1)
        b_hat = torch.stack((bx_hat.view(-1), by_hat.view(-1)))#.permute(2, 0, 1)

        return torch.mean(torch.abs(b_hat))

class detDFunc(nn.Module):
    def __init__(self, device, size):
        super(detDFunc, self).__init__()
        self.detD_loss = detD(device, *size)

    def forward(self, pred):
        return self.detD_loss(pred)

def detD(device, H, W):
    face, vertex, boundary_mask = meshgen(H, W, return_boundary_mask = True)
    #print('mask', boundary_mask, boundary_mask.shape, np.sum(boundary_mask))
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)
    boundary_mask = torch.from_numpy(boundary_mask[np.newaxis, ...]).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]

    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]

    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi

    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    relu = nn.ReLU()

    def detD_loss(mapping):
        """
        Inputs:
            mapping: (N, 2, h, w), torch tensor
        Outputs:
            loss: (N, (h-1)*(w-1)*2), torch tensor
        """
        N, C, H, W = mapping.shape
        mapping = mapping.reshape((N, C, -1))
        mapping = mapping.permute(0, 2, 1)

        si = mapping[:, face[:, 0], 0]
        sj = mapping[:, face[:, 1], 0]
        sk = mapping[:, face[:, 2], 0]

        ti = mapping[:, face[:, 0], 1]
        tj = mapping[:, face[:, 1], 1]
        tk = mapping[:, face[:, 2], 1]

        sjsi = sj - si
        sksi = sk - si
        tjti = tj - ti
        tkti = tk - ti

        a = (sjsi * hkhi + sksi * hihj) / area / 2;
        b = (sjsi * gigk + sksi * gjgi) / area / 2;
        c = (tjti * hkhi + tkti * hihj) / area / 2;
        d = (tjti * gigk + tkti * gjgi) / area / 2;

        det = (a*d-c*b) * boundary_mask
        loss = torch.mean(relu(-det))
        return loss #mu
    return detD_loss


