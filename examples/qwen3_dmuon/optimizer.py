"""Direct-space Muon recipe with owner-local momentum and FP16 polynomial steps."""

import torch

POLAR = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
COEFFICIENTS = tuple((a / 1.05, b / 1.05**3, c / 1.05**5) for a, b, c in POLAR)


def orthogonalize(gradient):
    """Apply dmuon's direct-space polynomial, with its normalization and dtype flow."""
    x = gradient.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = (x / (x.norm() + 1e-7)).half()
    for a, b, c in COEFFICIENTS:
        gram = x @ x.T
        polynomial = b * gram + (c * gram) @ gram
        x = a * x + polynomial @ x
    return (x.T if transpose else x).to(gradient.dtype)


@torch.no_grad()
def step(weight, gradient, momentum=None, *, lr=0.02, mu=0.95, weight_decay=0.01):
    """Apply Nesterov momentum and the full-matrix Moonlight learning-rate scale."""
    momentum = gradient.clone() if momentum is None else mu * momentum + gradient
    update = orthogonalize(gradient + mu * momentum)
    scale = 0.2 * max(weight.shape) ** 0.5
    return weight * (1 - lr * weight_decay) - lr * scale * update, momentum
