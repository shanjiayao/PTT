import os

import torch
import torch.nn as nn

from .. import backbones_3d, voting_heads, similarity_modules


class Tracker3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.training = dataset.training
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'backbone_3d', 'similarity_module', 'centroid_voting_head', 'box_voting_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def forward(self, **kwargs):
        raise NotImplementedError

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(model_info_dict=model_info_dict)
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_similarity_module(self, model_info_dict):
        if self.model_cfg.get('SIMILARITY_MODULE', None) is None:
            return None, model_info_dict
        centroid_voting_head_module = similarity_modules.__all__[self.model_cfg.SIMILARITY_MODULE.NAME](
            model_cfg=self.model_cfg.SIMILARITY_MODULE
        )
        model_info_dict['module_list'].append(centroid_voting_head_module)
        return centroid_voting_head_module, model_info_dict

    def build_centroid_voting_head(self, model_info_dict):
        if self.model_cfg.get('CENTROID_HEAD', None) is None:
            return None, model_info_dict
        centroid_voting_head_module = voting_heads.__all__[self.model_cfg.CENTROID_HEAD.NAME](
            model_cfg=self.model_cfg.CENTROID_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=1,
        )
        model_info_dict['module_list'].append(centroid_voting_head_module)
        return centroid_voting_head_module, model_info_dict

    def build_box_voting_head(self, model_info_dict):
        if self.model_cfg.get('BOX_HEAD', None) is None:
            return None, model_info_dict
        centroid_voting_head_module = voting_heads.__all__[self.model_cfg.BOX_HEAD.NAME](
            model_cfg=self.model_cfg.BOX_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=1,
        )
        model_info_dict['module_list'].append(centroid_voting_head_module)
        return centroid_voting_head_module, model_info_dict

    def post_processing(self, batch_dict):
        pass

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            print('file path:', filename)
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    @staticmethod
    def calc_flops(model, input_dict):
        def conv1d_hook(self, input, output):
            batch_size, input_channels, input_length = input[0].size()
            output_channels, output_length = output[0].size()

            kernel_ops = self.kernel_size[0] * (self.in_channels /
                self.groups) * (2 if multiply_adds else 1)
            bias_ops = 1 if self.bias is not None else 0

            params = output_channels * (kernel_ops + bias_ops)
            flops = batch_size * params * output_length

            list_conv.append(flops)
            table.add_row([self, flops / 1e6 / 2])

        def conv2d_hook(self, input, output):
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()

            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
                2 if multiply_adds else 1)
            bias_ops = 1 if self.bias is not None else 0

            params = output_channels * (kernel_ops + bias_ops)
            flops = batch_size * params * output_height * output_width

            list_conv.append(flops)
            table.add_row([self, flops / 1e6 / 2])

        def conv3d_hook(self, input, output):
            batch_size, input_channels, input_height, input_width, input_length = input[0].size()
            output_channels, output_height, output_width, output_length = output[0].size()

            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (self.in_channels /
                self.groups) * (2 if multiply_adds else 1)
            bias_ops = 1 if self.bias is not None else 0

            params = output_channels * (kernel_ops + bias_ops)
            flops = batch_size * params * output_height * output_width * output_length

            list_conv.append(flops)
            table.add_row([self, flops / 1e6 / 2])

        def linear_hook(self, input, output):
            batch_size = input[0].size(0) if input[0].dim() == 2 else 1

            weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)

            if self.bias is not None:
                bias_ops = self.bias.nelement()
            else:
                bias_ops = 0

            flops = batch_size * (weight_ops + bias_ops)
            list_linear.append(flops)
            table.add_row([self, flops / 1e6 / 2])

        def bn_hook(self, input, output):
            list_bn.append(input[0].nelement())
            # table.add_row([self, input[0].nelement() / 1e6 / 2])

        def relu_hook(self, input, output):
            list_relu.append(input[0].nelement())
            # table.add_row([self, input[0].nelement() / 1e6 / 2])

        def pooling_hook(self, input, output):
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()

            kernel_ops = self.kernel_size * self.kernel_size
            bias_ops = 0
            params = output_channels * (kernel_ops + bias_ops)
            flops = batch_size * params * output_height * output_width

            list_pooling.append(flops)
            table.add_row([self, flops / 1e6 / 2])

        def foo(net):
            childrens = list(net.children())
            if not childrens:
                if isinstance(net, torch.nn.Conv1d):
                    net.register_forward_hook(conv1d_hook)
                if isinstance(net, torch.nn.Conv2d):
                    net.register_forward_hook(conv2d_hook)
                if isinstance(net, torch.nn.Conv3d):
                    net.register_forward_hook(conv3d_hook)
                if isinstance(net, torch.nn.Linear):
                    net.register_forward_hook(linear_hook)
                if isinstance(net, torch.nn.BatchNorm2d):
                    net.register_forward_hook(bn_hook)
                if isinstance(net, torch.nn.ReLU):
                    net.register_forward_hook(relu_hook)
                if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                    net.register_forward_hook(pooling_hook)
                return
            for c in childrens:
                foo(c)

        from prettytable import PrettyTable
        table = PrettyTable(["Modules", "FLOPs"])
        multiply_adds = False
        list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
        foo(model)
        _ = model(input_dict)

        total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
        print(table)
        # print(f"Total FLOPs: {total_flops}")
        print('Total FLOPs: %.2fM' % (total_flops / 1e6 / 2))
        print('do not include RELU')

    @staticmethod
    def count_parameters(model):
        from prettytable import PrettyTable
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params