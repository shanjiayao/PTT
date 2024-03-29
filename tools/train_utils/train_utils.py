import glob
import os
import torch
import tqdm

import tools.eval_utils.eval_tracking_utils as eval_utils
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    logger=None, log_interval=100, loss_record=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        if optim_cfg.get('SCHEDULER', None) is None:
            lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        if torch.cuda.get_device_name(0) == 'GeForce GTX 1660 Ti' and optim_cfg.DEBUG:
            for k, v in batch.items():
                try:
                    print(k, ':\t', v.shape)
                except AttributeError:
                    print(k, ':\t', v)

        loss, tb_dict, disp_dict = model_func(model, batch)
        loss.backward()
        if optim_cfg.get('GRAD_NORM_CLIP') is not None:
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        if loss_record:
            loss_record.update(loss.item())

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(cfg, logger, model, optimizer, train_loader, test_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, loss_record_handle=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if loss_record_handle:
                loss_record = loss_record_handle()
            else:
                loss_record = None

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                logger=logger,
                loss_record=loss_record
            )
            if optim_cfg.get('SCHEDULER', None) is not None:   # for scheduler like step
                lr_scheduler.step(cur_epoch)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # eval model in evaluation set
            if cfg.TRAIN.WITH_EVAL.ENABLE and \
                    trained_epoch >= cfg.TRAIN.WITH_EVAL.START_EPOCH and \
                    trained_epoch % cfg.TRAIN.WITH_EVAL.INTERVAL == 0 \
                    and rank == 0:
                logger.info('**********************Eval %d Start**********************' % trained_epoch)
                eval_output_dir = ckpt_save_dir.parent / 'eval' / ('epoch_%d' % trained_epoch) / \
                                  cfg.DATA_CONFIG.DATA_SPLIT['test']
                eval_output_dir.mkdir(parents=True, exist_ok=True)

                model.eval()
                with torch.no_grad():
                    eval_utils.eval_one_epoch(
                        cfg=cfg, model=model, dataloader=test_loader, epoch_id=trained_epoch, logger=logger,
                        dist_test=False, result_dir=eval_output_dir, save_to_file=False, tb_log=tb_log
                    )
                logger.info('**********************Eval %d Over**********************' % trained_epoch)


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': 'none'}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
