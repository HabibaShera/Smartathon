import torch
from torch import nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, x):
        return


def revert_sync_batchnorm(module):
    """
    :param module: nn.Module to modify the settings for inference
    :return: modified model
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                    module.eps, module.momentum,
                                    module.affine,
                                    module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):
    """
    creates a traced model where every operation done by the model is saved to a graph in Torch-script to
    be loaded into another backend
    """
    def __init__(self, model, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()
        self.model = revert_sync_batchnorm(model)
        self.model.to('cpu')
        self.model.eval()
        self.detect_layer = self.model.model[-1]
        self.model.traced = True

        rand_example = torch.rand(1, 3, *img_size)
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)

    def forward(self, inputs, augment=False):
        out = self.model(inputs)
        return self.detect_layer(out)

    def save(self, name):
        self.model.save(name)
        print(f'saved_{name}')


def load_model(chk_path, map_location=None):
    """
    :param chk_path:str path of the weights
    :param map_location:device to map to
    :return: the saved model
    """
    chk = torch.load(chk_path, map_location=map_location)
    model = chk['ema' if chk.get('ema') else 'model'].float().fuse().eval()
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    return model
