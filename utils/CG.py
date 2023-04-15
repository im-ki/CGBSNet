import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CG(A, init_guess = None):
    # shape of A: (N, 7, 112, 112), torch
    # shape init_guess = (N, 2, 112, 112), torch
    # shape of x, rx, px: (N, 1, 112, 110)
    # shape of y, ry, py: (N, 1, 110, 112)
    unfold = nn.Unfold((3, 3), stride=(1, 1), padding=(1, 1))

    N, _, h, w = A.shape
    device = A.device

    if init_guess is not None:
        x0 = init_guess
    else:
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        y = y[::-1, :]
        x = x / (w-1)
        y = y / (h-1)
        x0 = torch.from_numpy(np.array((x, y))[np.newaxis, :]).repeat((N, 1, 1, 1))
    x0 = x0.to(device = device)

    x0_unfold = unfold(x0).reshape((N, 2, 9, h, w))
    x = x0_unfold[:, 0, 1:-1, :, 1:-1]
    y = x0_unfold[:, 1, 1:-1, 1:-1, :]

    lm_mask = torch.zeros_like(x0).to(device = device)
    lm_mask[:, 0, :, 0] = 1
    lm_mask[:, 0, :, -1] = 1
    lm_mask[:, 1, 0, :] = 1
    lm_mask[:, 1, -1, :] = 1
    lm_mask_unfold = unfold(lm_mask).reshape((N, 2, 9, h, w))
    x_lm_mask_unfold = lm_mask_unfold[:, 0]
    y_lm_mask_unfold = lm_mask_unfold[:, 1]
    Ax_spd = (1-x_lm_mask_unfold[:, 1:-1, :, 1:-1]) * A[:, :, :, 1:-1]
    Ay_spd = (1-y_lm_mask_unfold[:, 1:-1, 1:-1, :]) * A[:, :, 1:-1, :]

    lm_mask = torch.zeros_like(x0).to(device = device)
    lm_mask[:, 0, :, 0] = 0
    lm_mask[:, 0, :, -1] = 1
    lm_mask[:, 1, 0, :] = 1
    lm_mask[:, 1, -1, :] = 0
    lm_mask_unfold = unfold(lm_mask).reshape((N, 2, 9, h, w))
    x_lm_mask_unfold = lm_mask_unfold[:, 0]
    y_lm_mask_unfold = lm_mask_unfold[:, 1]
    bx = - torch.sum(x_lm_mask_unfold[:, 1:-1, :, 1:-1] * A[:, :, :, 1:-1], dim=1, keepdim = True)
    by = - torch.sum(y_lm_mask_unfold[:, 1:-1, 1:-1, :] * A[:, :, 1:-1, :], dim=1, keepdim = True)

    rx = bx - torch.sum(Ax_spd * x, dim = 1, keepdim = True)
    ry = by - torch.sum(Ay_spd * y, dim = 1, keepdim = True)

    px = rx
    py = ry

    x = x0[:, 0, :, 1:-1].unsqueeze(1)
    y = x0[:, 1, 1:-1, :].unsqueeze(1)

    while True:
        with torch.no_grad():
            px_pad = F.pad(px, (1, 1), "constant", 0)
            py_pad = F.pad(py, (0, 0, 1, 1), "constant", 0)
            px_pad_unfold = unfold(px_pad).reshape((N, 9, h, w))
            py_pad_unfold = unfold(py_pad).reshape((N, 9, h, w))

            Apx = torch.sum(Ax_spd * px_pad_unfold[:, 1:-1, :, 1:-1], dim = 1, keepdim = True)
            Apy = torch.sum(Ay_spd * py_pad_unfold[:, 1:-1, 1:-1, :], dim = 1, keepdim = True)

            pApx = torch.sum(px * Apx, dim = (1, 2, 3))
            pApy = torch.sum(py * Apy, dim = (1, 2, 3))

            rrx = torch.sum(rx * rx, dim = (1, 2, 3))
            rry = torch.sum(ry * ry, dim = (1, 2, 3))

            alpha_x = rrx / pApx
            alpha_y = rry / pApy
            alpha_x = alpha_x.reshape((N, 1, 1, 1))
            alpha_y = alpha_y.reshape((N, 1, 1, 1))

            x = x + alpha_x * px
            y = y + alpha_y * py

            rx_next = rx - alpha_x * Apx
            ry_next = ry - alpha_y * Apy

            beta_x = torch.sum(rx_next**2, dim = (1, 2, 3)) / torch.sum(rx**2, dim = (1, 2, 3))
            beta_y = torch.sum(ry_next**2, dim = (1, 2, 3)) / torch.sum(ry**2, dim = (1, 2, 3))
            beta_x = beta_x.reshape((N, 1, 1, 1))
            beta_y = beta_y.reshape((N, 1, 1, 1))

            px_next = rx_next + beta_x * px
            py_next = ry_next + beta_y * py

            rx, ry = rx_next, ry_next
            px, py = px_next, py_next

        yield x, y, rx, ry#, px, py
