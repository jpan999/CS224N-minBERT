from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        print("AdamW, wd", weight_decay)
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # initialize the state
                # print(1)
                if len(state) == 0:
                    state['first_moment'] = torch.zeros_like(p.data)
                    state['second_moment'] = torch.zeros_like(p.data)
                    state['first_moment'] = state['first_moment'].to(p.device)
                    state['second_moment'] = state['second_moment'].to(p.device)
                    state['timestep'] = 0

                # read the state and hyperparameters
                first_moment, second_moment, timestep = state['first_moment'], state['second_moment'], state['timestep']
                beta_1, beta_2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                timestep += 1
                # 1- Update first and second moments of the gradients
                first_moment = torch.mul(first_moment, beta_1) + torch.mul(grad, 1-beta_1)
                second_moment = torch.mul(second_moment, beta_2) + torch.mul(grad * grad, 1-beta_2)

                # 2- Apply bias correction
                if group['correct_bias']:
                    numerator = 1 - beta_2 ** timestep
                    denominator = 1 - beta_1 ** timestep
                    new_alpha = alpha * math.sqrt(numerator) / denominator

                # 3- Update parameters (p.data).
                p.data = p.data - new_alpha * first_moment / (torch.sqrt(second_moment) + eps)

                # 4- After that main gradient-based update, update again using weight decay
                if weight_decay > 0:
                    p.data = p.data - alpha * weight_decay * p.data

                # save the state
                state['timestep'] = timestep
                state['first_moment'] = first_moment
                state['second_moment'] = second_moment

        return loss
