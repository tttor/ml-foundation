#!/usr/bin/env python3
import torch

def main():
    n = 3; torch.manual_seed(123)

    mean_1 = torch.rand(n)
    stddev_1 = torch.rand_like(mean_1)
    normal_1 = torch.distributions.Normal(loc=mean_1, scale=stddev_1)

    mean_2 = torch.rand(n, requires_grad=True)
    stddev_2 = torch.rand_like(mean_2, requires_grad=True)
    normal_2 = torch.distributions.Normal(loc=mean_2, scale=stddev_2)

    # Hellinger
    print('Hellinger ---')
    dist = hellinger(normal_1, normal_2).mean()

    if mean_2.grad is not None: mean_2.grad.data.zero_()
    if stddev_2.grad is not None: stddev_2.grad.data.zero_()
    dist.backward()

    print('mean_2.grad=', mean_2.grad)
    print('stddev_2.grad=', stddev_2.grad)

    # KL-div
    print('KL-div ---')
    dist = torch.distributions.kl.kl_divergence(normal_1, normal_2).mean()

    if mean_2.grad is not None: mean_2.grad.data.zero_()
    if stddev_2.grad is not None: stddev_2.grad.data.zero_()
    dist.backward()

    print('mean_2.grad=', mean_2.grad)
    print('stddev_2.grad=', stddev_2.grad)

def hellinger(normal_1, normal_2):
    # https://en.wikipedia.org/wiki/Hellinger_distance
    mu_1 = normal_1.loc; sigma_1 = normal_1.scale
    mu_2 = normal_2.loc; sigma_2 = normal_2.scale

    sqrt_term = (2*sigma_1*sigma_2) / (sigma_1**2 + sigma_2**2)
    exp_term = -0.25 * (mu_1 - mu_2)**2 / (sigma_1**2 + sigma_2**2)
    hel_squared = 1. - (sqrt_term.sqrt() * exp_term.exp())

    return hel_squared.sqrt()

if __name__ == '__main__':
    main()
