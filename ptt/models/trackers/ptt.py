# -*- coding: utf-8 -*-
from .tracker3d_template import Tracker3DTemplate


"""
█████╗  ████████╗███████╗
██╔══██╗╚══██╔══╝╚══██╔══╝
███████║   ██║      ██║
██╔════║   ██║      ██║
██║        ██║      ██║
╚═╝        ╚═╝      ╚═╝
"""


class PTT(Tracker3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                search_points :  torch.Size([1024, 4])
                template_points :        torch.Size([512, 4])
                batch_size :     1
                search_seeds :   torch.Size([128, 4])
                search_feats :   torch.Size([128, 256])
                search_inds :    torch.Size([1, 128])
                template_seeds :         torch.Size([64, 4])
                template_feats :         torch.Size([64, 256])
                template_inds :  torch.Size([1, 64])
                cosine_feats :   torch.Size([128, 256])
                pred_centroids_cls :     torch.Size([128])
                pred_centroids_votes :   torch.Size([1, 128, 3])
                votes_feats :    torch.Size([128, 257])
                pred_box_center :        torch.Size([1, 64, 3])
                pred_box_data :  torch.Size([1, 64, 5])
        Returns:

        """
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss.float()
            }
            return ret_dict, tb_dict, disp_dict
        else:
            return batch_dict

    def get_training_loss(self):
        disp_dict = {}
        loss_centroids, tb_dict = self.centroid_voting_head.get_loss()
        loss_boxes, tb_dict = self.box_voting_head.get_loss(tb_dict)

        loss = loss_centroids + loss_boxes
        disp_dict.update(tb_dict)
        return loss, tb_dict, disp_dict
