"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import jittor as jt
import jittor.nn as nn

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0
jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight = jt.float16(l.weight.data)
        # l.weight.assign(l.weight.float16())
        if l.bias is not None:
            # l.bias.assign(l.bias.float16())
            l.bias = jt.float16(l.bias.data)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            # _flatten_dense_tensors(
            #     [param.detach().float() for (_, param) in param_group]
            # ).view(shape)
            jt.cat(
                [param.detach().float().contiguous().view(-1) for (_, param) in param_group], dim=0
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):  
            param.detach()
            param = unflat_master_param.copy()


def unflatten_master_params(param_group, master_param):
    # TODO:
    unflatten_list = []
    start_pos = 0
    for (_, param) in param_group:
        unflatten_list.append(master_param[start_pos : start_pos + param.numel()].view(param.shape))
        start_pos += param.numel()
    return unflatten_list


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        opt,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.opt = opt
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def model_grads_to_master_grads(self, param_groups_and_shapes, master_params):
        """
        Copy the gradients from the model parameters into the master parameters
        from make_master_params().
        """
        for master_param, (param_group, shape) in zip(
            master_params, param_groups_and_shapes
        ):
            self.opt._grad_map[id(master_param)] = jt.cat(
                    [param.detach().float().contiguous().view(-1) for (_, param) in param_group], dim=0
                ).view(shape)

    def zero_grad(self):
        for param in self.model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.opt_grad(self.opt) is not None:
                param.opt_grad(self.opt).detach_inplace()
                param.opt_grad(self.opt).zero_()

    def zero_master_grads(self, master_params):
        for param in master_params:
            # param.opt_grad(optimizer) = None
            self.opt._grad_map[id(param)] = None

    def param_grad_or_zeros(self, param):
        if param.opt_grad(self.opt) is not None:
            return param.opt_grad(self.opt).data.detach()
        else:
            return jt.zeros_like(param)

    def backward(self, loss: jt.Var):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            self.opt.backward(loss * loss_scale)
        else:
            self.opt.backward(loss)

    def optimize(self, opt: jt.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: jt.optim.Optimizer):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        self.model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            self.zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].opt_grad(opt).mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        self.zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: jt.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with jt.no_grad():
                param_norm += jt.norm(p, p=2, dim=list(range(len(p.shape)))).item()** 2
                grad = p.opt_grad(self.opt)
                if grad is not None:
                    grad_norm += jt.norm(grad, p=2, dim=list(range(len(grad.shape)))).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
