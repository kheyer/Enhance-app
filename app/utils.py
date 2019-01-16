#import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *

from torchvision.models import vgg16_bn

vgg_m = vgg16_bn(True).features.eval()
requires_grad(vgg_m, False)

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]

def calc_2_moments(tensor):
    chans = tensor.shape[1]
    tensor = tensor.view(1, chans, -1)
    n = tensor.shape[2]
    mu = tensor.mean(2)
    tensor = (tensor - mu[:,:,None]).squeeze(0)
    cov = torch.mm(tensor, tensor.t()) / float(n)

    return mu, cov

def get_style_vals(tensor):
    mean, cov = calc_2_moments(tensor)
    eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
    eigroot_mat = torch.diag(torch.sqrt(eigvals.clamp(min=0)))
    root_cov = torch.mm(torch.mm(eigvects, eigroot_mat), eigvects.t())
    tr_cov = eigvals.clamp(min=0).sum()
    return mean, tr_cov, root_cov

def calc_l2wass_dist(mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth):
    tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
    mean_diff_squared = (mean_stl - mean_synth).pow(2).sum()
    cov_prod = torch.mm(torch.mm(root_cov_stl, cov_synth), root_cov_stl)
    var_overlap = torch.sqrt(torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0)+1e-8).sum()
    dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2*var_overlap
    return dist

def single_wass_loss(pred, targ):
    mean_test, tr_cov_test, root_cov_test = targ
    mean_synth, cov_synth = calc_2_moments(pred)
    loss = calc_l2wass_dist(mean_test, tr_cov_test, root_cov_test, mean_synth, cov_synth)
    return loss

class FeatureLoss_Wass(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts, wass_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.wass_wgts = wass_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'wass_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        styles = [get_style_vals(i) for i in out_feat]
        self.feat_losses += [single_wass_loss(f_pred, f_targ)*w
                            for f_pred, f_targ, w in zip(in_feat, styles, self.wass_wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()
