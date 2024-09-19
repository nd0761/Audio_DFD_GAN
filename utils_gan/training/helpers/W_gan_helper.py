import torch
# https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/blob/master/trainer.py
def calculate_gradient_penalty(model, real, noised, device): #model=disc
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    # print(real.shape)
    alpha = torch.randn((real.size(0), 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real + ((1 - alpha) * noised)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return ((gradients_norm - 1) ** 2).mean()