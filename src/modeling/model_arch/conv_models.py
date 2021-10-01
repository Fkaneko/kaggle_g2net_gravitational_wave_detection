from typing import Callable, Dict, List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.modeling.model_arch.smp_untepp_wo_pool import UnetPlusPlus

TIMM_CHANNELS = {
    "efficientnet_b0": 1280,
    "efficientnet_b3a": 1536,
    "tf_efficientnet_b6": 2304,
    "tf_efficientnet_b7": 2560,
}

SMP_MODELS = {
    "unet": smp.Unet,
    "unetpp": UnetPlusPlus,  # import different model from official smp
    "manet": smp.MAnet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3p": smp.DeepLabV3Plus,
}


# GeM from
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.2/cirtorch/layers/pooling.py#L36
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad: bool = True):
        super().__init__()
        self.p = nn.Parameter(data=torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class SegConv(nn.Module):
    def __init__(
        self,
        smp_params: DictConfig,
        num_classes: int = 1,
    ):
        super().__init__()
        if isinstance(smp_params, DictConfig):
            smp_params = OmegaConf.to_container(smp_params)
        smp_params["classes"] = num_classes
        smp_arch = smp_params.pop("arch_name")
        smp_net = SMP_MODELS[smp_arch]
        self.seg_net = smp_net(**smp_params)
        self.smp_params = smp_params

    def forward(self, inputs: torch.Tensor, aux_inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.seg_net(inputs)
        if hasattr(self.smp_params, "aux_params"):
            outputs, labels = outputs
        else:
            labels = None
        return {"outputs": outputs, "aux_out": labels}


class SegDenoise(SegConv):
    def __init__(
        self,
        img_mean: torch.Tensor,
        img_std: torch.Tensor,
        smp_params: DictConfig,
        num_sites: int = 3,
        crop_size: Tuple[int, int] = (128, 128),
        noise_scaler: float = 1.0,
        predict_mask: bool = False,
        timm_params: DictConfig = None,
        gem_power: int = 3,
        gem_requires_grad: bool = False,
        use_full_encoder: bool = False,
        use_cam: bool = True,
        skip_max_pool: bool = True,
    ):
        super().__init__(smp_params=smp_params, num_classes=num_sites)
        self.crop_size = crop_size
        # self.noise_scaler = noise_scaler
        self.noise_act = nn.Tanh()
        self.num_sites = num_sites
        self.predict_mask = predict_mask
        self.smp_params = smp_params
        self.use_full_encoder = use_full_encoder
        self.use_cam = use_cam
        self.skip_max_pool = skip_max_pool

        if self.use_cam:
            assert hasattr(smp_params, "aux_params")
        self._img_mean = nn.Parameter(data=img_mean, requires_grad=False)
        self._img_std = nn.Parameter(data=img_std, requires_grad=False)

        if hasattr(smp_params, "aux_params"):
            if use_cam:
                self.aux_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # for CAM
            else:
                self.aux_pool = GeM(p=gem_power, requires_grad=gem_requires_grad)

            if self.use_full_encoder:
                timm_model_names = smp_params.encoder_name.replace("timm-", "").replace(
                    "-", "_"
                )
                self.enccoder_in_features = TIMM_CHANNELS[timm_model_names]
            else:
                self.enccoder_in_features = 320
            self.aux_head = nn.Linear(self.enccoder_in_features, 1)

        if predict_mask:
            self.mask_cast = nn.Conv2d(
                in_channels=num_sites,
                out_channels=2,  # value and mask
                kernel_size=1,
                stride=1,
                bias=True,
                padding="same",
            )
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            # self.bn = nn.BatchNorm2d(num_sites)
            self.std_scaler = nn.PReLU(num_parameters=num_sites)

    def forward(
        self,
        inputs: Dict[str, Union[torch.Tensor, str]],
        aux_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = inputs[:, :, : self.crop_size[0], : self.crop_size[1]]
        noise_mask = self.seg_net(inputs)
        if hasattr(self.smp_params, "aux_params"):
            noise_mask, feat_map = noise_mask
            if self.use_full_encoder:
                feat_map = self.seg_net.encoder.conv_head(feat_map)
                feat_map = self.seg_net.encoder.bn2(feat_map)
                feat_map = self.seg_net.encoder.act2(feat_map)
            aux_out = self.aux_pool(feat_map).squeeze()
            aux_out = self.aux_head(aux_out)
        else:
            aux_out = None

        noise_mask = self.noise_act(noise_mask)

        inputs = inputs * self._img_std + self._img_mean
        # inputs = gt + noise
        outputs = inputs[:, : self.num_sites] - noise_mask

        if self.use_cam:
            cam = F.conv2d(
                feat_map, self.aux_head.weight.view(1, self.enccoder_in_features, 1, 1)
            )
            cam = cam / (F.adaptive_max_pool2d(cam, output_size=(1, 1)) + 1.0e-5)
            cam *= torch.sigmoid(aux_out).view(-1, 1, 1, 1)
            cam = F.interpolate(
                cam, self.crop_size, mode="bilinear", align_corners=False
            )
            outputs = outputs + cam

        if self.predict_mask:
            # mask generation
            outputs = self.mask_cast(outputs)
            if self.skip_max_pool:
                outputs[:, 1:2] = torch.sigmoid(outputs[:, 1:2])
            else:
                outputs[:, 1:2] = self.max_pool(torch.sigmoid(outputs[:, 1:2]))
        else:
            outputs = self.std_scaler(outputs)
        return {"outputs": outputs, "aux_out": aux_out}


class SegAct(nn.Module):
    def __init__(self, predict_mask: bool = True, scaler: float = 10.0):
        super().__init__()
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.predict_mask = predict_mask

    def forward(self, logits: torch.Tensor, is_act: bool = True):
        assert logits.ndim == 4

        # pred_vaule/mask pred
        if self.predict_mask:
            # logits = torch.sigmoid(logits)
            # move to model
            # mask = self.max_pool(torch.sigmoid(logits[:, 1:2]))
            mask = logits[:, 1:2]
            outputs = (logits[:, 0:1] * mask) / (
                mask.sum(dim=(-3, -2, -1), keepdim=True) + 1.0e-6
            )
            outputs = outputs.sum(dim=(-3, -2, -1)).squeeze()
            if is_act:
                return torch.sigmoid(outputs)
            else:
                return outputs
        else:
            if is_act:
                return logits.std(dim=(-2, -1)).mean(dim=-1)
            else:
                return logits


class TimmConv(nn.Module):
    def __init__(
        self,
        timm_params: DictConfig,
        num_classes: int = 1,
        gem_power: int = 3,
        gem_requires_grad: bool = True,
    ):
        super().__init__()

        if isinstance(timm_params, DictConfig):
            timm_params = OmegaConf.to_container(timm_params)

        self.backbone = timm.create_model(
            timm_params["encoder_name"],
            pretrained=timm_params["pretrained"],
            num_classes=0,  # define later
            global_pool="",  # define later
        )

        self.pool = GeM(p=gem_power, requires_grad=gem_requires_grad)

        self.head = nn.Linear(TIMM_CHANNELS[timm_params["encoder_name"]], 1)

    def forward(
        self,
        inputs: torch.Tensor,
        aux_inputs: Optional[torch.Tensor] = None,
    ):

        x = self.backbone(inputs)
        x = self.pool(x).squeeze()
        x = self.head(x)

        return {"outputs": x, "aux_out": None}


class SiteFuse(nn.Module):
    def __init__(
        self,
        fuse_type: str = "pooled_concat",
        num_features: int = 1380,
        num_sites: int = 3,
        out_channels: int = 512,
    ):
        super().__init__()
        self.fuse_type = fuse_type

        if self.fuse_type == "pooled_concat":
            # (ba, 3 x features)
            self.fuse_module = nn.Sequential(
                nn.Linear(num_features * num_sites, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.PReLU(),
            )
        elif self.fuse_type == "pooled_ch_stack":
            # (ba, 3, features)
            pass
        elif self.fuse_type == "pooled_seq_stack":
            # (ba, features, 3)
            self.fuse_module = nn.Sequential(
                nn.Conv1d(
                    in_channels=num_features,
                    out_channels=out_channels,
                    kernel_size=num_sites,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm1d(out_channels),
                nn.PReLU(),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm1d(out_channels),
                nn.PReLU(),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    bias=False,
                    padding="valid",
                ),
                nn.BatchNorm1d(out_channels),
                nn.PReLU(),
            )

        elif self.fuse_type == "feature_concat":
            # (ba, 3*features, w, h)
            self.fuse_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_features * num_sites,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )

    def forward(
        self, features: List[torch.Tensor], pool_func: Callable
    ) -> torch.Tensor:

        if self.fuse_type == "pooled_concat":
            # (ba, 3 x features)
            x = torch.cat(features, axis=-1)
        elif self.fuse_type == "pooled_ch_stack":
            # (ba, 3, features)
            x = torch.stack(features, axis=1)
        elif self.fuse_type == "pooled_seq_stack":
            # (ba, features, 3)
            x = torch.stack(features, axis=-1)
        elif self.fuse_type == "feature_concat":
            # (ba, 3*features, w, h)
            x = torch.cat(features, axis=1)

        x = self.fuse_module(x)
        if x.ndim == 4:
            x = pool_func(x)

        return x.squeeze()


class SiteSplitConv(TimmConv):
    def __init__(
        self,
        timm_params: DictConfig,
        channels_per_site: int = 3,
        norm_feature: bool = True,
        num_sites: int = 3,
        num_classes=1,
        in_channels=1,
        gem_power: int = 3,
        gem_requires_grad: bool = True,
        fuse_type: str = "pooled_concat",
        out_channels: int = 512,
    ):
        super().__init__(
            timm_params=timm_params,
            gem_power=gem_power,
            gem_requires_grad=gem_requires_grad,
            num_classes=num_classes,
        )
        self.channels_per_site = channels_per_site
        self.norm_feature = norm_feature

        # self.fuse1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=1)

        self.fuse_type = fuse_type
        self.fuse = SiteFuse(
            fuse_type=fuse_type,
            num_features=TIMM_CHANNELS[timm_params["encoder_name"]],
            num_sites=num_sites,
            out_channels=out_channels,
        )
        self.head = nn.Linear(out_channels, num_classes)

    @staticmethod
    def site_split(inputs: torch.Tensor, channels_per_site: int = 3):
        site_0 = inputs[:, 0::channels_per_site]
        site_1 = inputs[:, 1::channels_per_site]
        site_2 = inputs[:, 2::channels_per_site]
        return site_0, site_1, site_2

    def forward(
        self,
        inputs: torch.Tensor,
        aux_inputs: Optional[torch.Tensor] = None,
    ):
        site_inputs = SiteSplitConv.site_split(
            inputs=inputs, channels_per_site=self.channels_per_site
        )
        features = []
        for site_input in site_inputs:
            x = self.backbone(site_input)
            if self.fuse_type.startswith("pooled"):
                x = self.pool(x).squeeze()
            if self.norm_feature:
                x = F.normalize(x, p=2.0, dim=1)
            features.append(x)

        x = self.fuse(features, pool_func=self.pool)
        outputs = self.head(x)
        return {"outputs": outputs, "aux_out": None}


class PsdDenoise(TimmConv):
    def __init__(
        self,
        timm_params: DictConfig,
        gem_power: int = 3,
        gem_requires_grad: bool = True,
        num_sites: int = 3,
        num_features_per_site: int = 512,
        num_classes: int = 1,
    ):
        super().__init__(
            timm_params=timm_params,
            gem_power=gem_power,
            gem_requires_grad=gem_requires_grad,
            num_classes=num_classes,
        )
        self.num_sites = num_sites
        self.num_features_per_site = num_features_per_site
        self.feat_split_size = TIMM_CHANNELS[timm_params["encoder_name"]] // num_sites

        if (self.num_features_per_site - self.feat_split_size) > 0:
            self.pad = nn.ConstantPad1d(
                (0, self.num_features_per_site - self.feat_split_size), 0.0
            )
            in_feats = self.num_features_per_site
        else:
            in_feats = self.feat_split_size

        self.site_features_fix = nn.Sequential(
            nn.Linear(in_feats, self.num_features_per_site),
            nn.Tanh(),  # aux_inputs range 0 < x < 1
        )
        self.head = nn.Linear(self.num_features_per_site, num_classes)

    def forward(self, inputs: torch.Tensor, aux_inputs: torch.Tensor):
        x = self.backbone(inputs)
        x = self.pool(x).squeeze()
        site_features = torch.split(
            tensor=x[:, : self.feat_split_size * self.num_sites],
            split_size_or_sections=self.feat_split_size,
            dim=1,
        )
        denoised = []
        for site_ind, site_feature in enumerate(site_features):
            if (self.num_features_per_site - self.feat_split_size) > 0:
                site_feature = self.pad(site_feature)
            site_feature = self.site_features_fix(site_feature)
            # in_psd(aux_inputs) = true psd(unknown)  + noise(model_prediction)
            denoised.append(aux_inputs[:, site_ind] - site_feature)

        # fuse
        x = torch.mean(torch.stack(denoised, axis=1), axis=1)
        #
        outputs = self.head(x.squeeze())
        return {"outputs": outputs, "aux_out": None}
