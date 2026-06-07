import torch
import torch.nn.functional as F


def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape
        probs = F.softmax(cZ * lamd, dim=1).T

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)
            c_new = inv_N_samples / (r.T @ probs).T
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        probs *= c.squeeze()
        probs = probs.T
        probs *= r.squeeze()

        return probs * N_samples