import torch
from torch.optim import Optimizer

class ModifAdagrad(Optimizer):
    def __init__(self, params, lr, mr):
        defaults = dict(lr=lr)
        super(ModifAdagrad, self).__init__(params, defaults)
        self.g = 0
        self.mr = mr

    @torch.no_grad()
    def step(self, closure=None):
        grad_norm = 0
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Обчислюємо норму градієнту
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                param_norm = param.grad.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        self.g = self.mr * self.g + grad_norm

        # Оновлення параметрів
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                param -= group['lr'] * param.grad / self.g

        return loss
