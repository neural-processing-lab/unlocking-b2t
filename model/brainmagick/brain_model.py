import torch
from torch import nn
from torch.nn import functional as F
import typing as tp

from model.brainmagick.model_utils import ConvSequence, SubjectLayers, ChannelMerger
from model.dataset_projector import DatasetProjector, SharedProjector

'''
The model is composed of a spatial attention layer, then a 1x1 convolution without activation. 
A Subject Layer is selected based on the subject index s, which consists in a 1x1 convolution learnt 
only for that subject with no activation. Then, we apply five convolutional blocks made of three convolutions. 
The first two use residual skip connection and increasing dilation, followed by a BatchNorm layer 
and a GELU activation. The third convolution is not residual, and uses a GLU activation (which halves 
the number of channels) and no normalization. Finally, we apply two 1x1 convolutions with a GELU in between.
'''
class BrainModel(nn.Module):
    # in_channels, out_channels, n_subjects all passed in from dataset
    def __init__(self, in_channels, out_channels, n_subjects, dataset, har_type): #dset1_subjects=None, dset2_subjects=None
        super(BrainModel, self).__init__()

        self.har_type = har_type
        
        # spatial attention layer
        if self.har_type == "gating":
            print("Using gating")
            num_datasets = len(dataset.split("_"))
            self.projector = DatasetProjector(sensor_dim=in_channels, model_dim=270, num_datasets=num_datasets)
            in_channels = 270
        elif self.har_type == "spatial_attention":
            print("Using spatial attention")
            merger_channels = 270
            self.spatial_attention = ChannelMerger(
                    merger_channels, pos_dim=2048, dropout=0.2,
                    usage_penalty=0.) 
            in_channels = merger_channels
        elif self.har_type == "both":
            # Use both spatial attention and dataset projector
            # For datasets with spatial information, SA projects to 270 channels, then padded up to 306 before projector
            # For datasets without spatial information, padded up to 306 before projector and projected down to 270 channels
            print("Using both spatial attention and dataset projector")
            num_datasets = len(dataset.split("_"))
            merger_channels = 270
            self.spatial_attention = ChannelMerger(
                merger_channels, pos_dim=2048, dropout=0.2, usage_penalty=0.
            )
            self.projector = DatasetProjector(
                sensor_dim=in_channels, model_dim=merger_channels, num_datasets=num_datasets
            )
            in_channels = merger_channels
        elif self.har_type == "padding":
            # Simple padding baseline that pads inputs < in_channels to in_channels
            print("Using padded shared projector")
            merger_channels = 270
            self.projector = SharedProjector(sensor_dim=in_channels, model_dim=merger_channels)
            in_channels = merger_channels


        # 1x1 convolution (no activation)
        self.init_conv = nn.Conv1d(in_channels, 270, 1)

        # subject layer (1x1 convolution specific to subject, no activation)
        self.n_subjects = n_subjects
        self.subject_layer = SubjectLayers(270, 270, n_subjects, False)

        # compute the sequences of channel sizes (always "meg")
        sizes = [270]
        sizes += [int(round(320 * 1. ** k)) for k in range(10)]

        # 5 conv blocks of:
            # conv with residual skip connection + increasing dilation
            # BatchNorm layer + GELU activation
            # conv with residual skip connection + increasing dilation
            # BatchNorm layer + GELU activation
            # conv (not residual) + GLU activation (no norm)
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=3, stride=1,
                      leakiness=0.0, dropout=0.0, dropout_input=0.0,
                      batch_norm=True, dilation_growth=2, groups=1,
                      dilation_period=5, skip=True, post_skip=False, scale=None,
                      rewrite=False, glu=2, glu_context=1, glu_glu=True,
                      activation=nn.GELU)
        self.conv_blocks = ConvSequence(sizes, **params)

        # final 1x1 conv + GELU activation + 1x1 conv
        final_channels = sizes[-1]
        self.final_convs = nn.Sequential(
            nn.Conv1d(final_channels, 2 * final_channels, 1),
            nn.GELU(),
            nn.ConvTranspose1d(2 * final_channels, out_channels, 3, 1, 0))

    def forward(self, inputs, subjects, sensor_xyz, dataset_id, functional=False):

        length = inputs.shape[-1]  # length of any of the inputs

        x = inputs

        if self.har_type == "spatial_attention":
            x = self.spatial_attention(x, sensor_xyz)

        if self.har_type == "gating":
            x = self.projector(x, dataset_id)

        if self.har_type == "both":
            if not torch.isnan(sensor_xyz).any():
                x = self.spatial_attention(x, sensor_xyz)
            x = self.projector(x, dataset_id)

        if self.har_type == "padding":
            x = self.projector(x)
        
        x = self.init_conv(x)

        x = self.subject_layer(x, subjects)

        x = self.conv_blocks(x, functional=functional)

        x = self.final_convs(x)

        assert x.shape[-1] >= length
        out = x[:, :, :length]
        return out