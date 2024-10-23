import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops.pairwise_distance import pairwise_distance
from geotransformer.utils.pointcloud import pc_normalize,apply_transform_tensor
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
)

from score.backbone import KPConvFPN

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

    def forward(self, data_dict):

        # Downsample point clouds
        feats = data_dict['features'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        points_c = data_dict['points'][-1].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]

        # 2. KPFCNN Encoder
        feats_c = self.backbone(feats, data_dict)

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        ) # 1,128,256
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        return ref_feats_c_norm,src_feats_c_norm

class Scorer(nn.Module):
    def __init__(self, cfg):
        super(Scorer, self).__init__()
        self.num_classes = cfg.model.num_classes

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GroupNorm(cfg.model.group_norm,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.GroupNorm(cfg.model.group_norm,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, data_dict, ref_feats_c_norm,src_feats_c_norm, trans):
        ref_length_c = data_dict['lengths'][-1][0].item()
        points_c = data_dict['points'][-1].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        src_points_c = apply_transform_tensor(src_points_c, trans)

        ref_points_c_norm,src_points_c_norm = pc_normalize(ref_points_c,src_points_c)

        # 1. get the nearest point's spatial dist and feature dist
        dist_mat = torch.exp(-pairwise_distance(ref_points_c_norm,src_points_c_norm))
        match_idx_ref = dist_mat.max(dim=1)[1] # index of each ref_point's NN point in src (Returns the index of min in each row)
        match_idx_src = dist_mat.max(dim=0)[1] # index of each src_point's NN point in ref (Returns the index of min in each column)
        min_dist_ref = dist_mat.max(dim=1)[0] # distance of ref_point to its NN point in src
        min_dist_src = dist_mat.max(dim=0)[0] # distance of src_point to its NN point in ref
        feat_match_score = torch.matmul(ref_feats_c_norm, src_feats_c_norm.transpose(-1, -2))
        # feat_match_score = torch.exp(-pairwise_distance(ref_feats_c_norm, src_feats_c_norm, normalized=True)) # feature similarity (0-1)
        feat_match_idx_ref = feat_match_score.max(dim=1)[1]
        feat_match_idx_src = feat_match_score.max(dim=0)[1]
        feat_match_ref = feat_match_score.max(dim=1)[0]
        feat_match_src = feat_match_score.max(dim=0)[0]
        feat_score_ref = feat_match_score[range(feat_match_score.shape[0]), match_idx_ref] # feature matching score of ref_point to its NN point in src
        feat_score_src = feat_match_score[match_idx_src, range(feat_match_score.shape[0])] # feature matching score of src_point to its NN point in ref
        dist_ref = dist_mat[range(dist_mat.shape[0]),feat_match_idx_ref]
        dist_src = dist_mat[feat_match_idx_src,range(dist_mat.shape[0])]

        # 2. classifier
        # Spatial Distance Nearest Points and Corresponding Feature Scores
        min_dist = torch.cat([min_dist_ref,min_dist_src])
        feat_score = torch.cat([feat_score_ref,feat_score_src])
        sorted_min_dist, perm_d = torch.sort(min_dist,descending=True)
        sorted_feat_dist = feat_score[perm_d]
        d_m = sorted_min_dist*sorted_feat_dist
        # Highest point of feature similarity and corresponding spatial distance
        match_feat = torch.cat([feat_match_ref,feat_match_src])
        match_dist = torch.cat([dist_ref,dist_src])
        sorted_match_feat, perm_f = torch.sort(match_feat,descending=True)
        sorted_match_dist = match_dist[perm_f]
        m_d = sorted_match_dist*sorted_match_feat

        geo_out = torch.cat([d_m,m_d])
        cls_logits = self.cls_head(geo_out.unsqueeze(0))
        return cls_logits


def create_geo_model(config):
    model = GeoTransformer(config)
    return model

def create_score_model(config):
    model = Scorer(config)
    return model

def main():
    from config import make_cfg
    cfg = make_cfg()
    geo_model = create_geo_model(cfg)
    score_model = create_score_model(cfg)
    print(geo_model.state_dict().keys())
    print(geo_model)
    print(score_model.state_dict().keys())
    print(score_model)


if __name__ == '__main__':
    main()