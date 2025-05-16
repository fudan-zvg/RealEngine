import torch
import torch.nn.functional as F
from submodules.GSLiDAR.utils.general_utils import inverse_sigmoid

class RayDropPrior(torch.nn.Module):
    def __init__(self, h, w):
        super().__init__()
        init = inverse_sigmoid(0.1 * torch.ones([1, h, w * 2]))
        self.lidar_raydrop_prior = torch.nn.Parameter(init, requires_grad=True)
        self.lr = None

    def capture(self):
        return (
            self.lidar_raydrop_prior,
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args=None):
        self.lidar_raydrop_prior, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

    def training_setup(self, training_args):
        if self.lr is None:
            self.lr = training_args.raydrop_prior_lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-15)

    def forward(self, towards):
        w = self.lidar_raydrop_prior.shape[-1] // 2
        if towards == "forward":
            lidar_raydrop_prior_from_envmap = self.lidar_raydrop_prior[:, :, :w]
        elif towards == "backward":
            lidar_raydrop_prior_from_envmap = self.lidar_raydrop_prior[:, :, w:]
        else:
            raise NotImplementedError(towards)
        return torch.sigmoid(lidar_raydrop_prior_from_envmap)

    def upscale(self, h, w):
        self.lidar_raydrop_prior = torch.nn.Parameter(F.interpolate(self.lidar_raydrop_prior[None], size=(h, w * 2),
                                                                    mode='bilinear', align_corners=True)[0])
        self.training_setup(None)
