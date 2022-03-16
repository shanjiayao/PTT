import argparse
import tqdm

from ptt.config import cfg, cfg_from_yaml_file
from ptt.datasets import build_dataloader
from ptt.models import build_network, load_data_to_gpu, model_fn_decorator
from ptt.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/ptt_best.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=None,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo-------------------------')

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=0,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    print(model)
    model.count_parameters(model)

    model.train()
    model.cuda()
    logger.info(f'Total number of samples: \t{len(train_set)}')

    with tqdm.trange(0, 100, desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        total_it_each_epoch = len(train_loader)
        logger.info(f'total_it_each_epoch: \t{len(train_loader)}')
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            batch = next(dataloader_iter)
            func = model_fn_decorator()
            load_data_to_gpu(batch)
            model.calc_flops(model, batch)
            loss, tb_dict, disp_dict = func(model, batch)
            import ipdb; ipdb.set_trace()
    exit()


if __name__ == '__main__':
    main()
