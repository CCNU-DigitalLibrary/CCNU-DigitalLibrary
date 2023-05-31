import datetime
import logging
import time

import numpy as np
import torch
import torch.distributed as dist

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import GradScaler, autocast
    amp_support = True
except:
    amp_support = False

from lib.utils.comm import get_world_size, synchronize, all_gather
from lib.utils.params import ContiguousParams
from lib.solver.build import make_lr_scheduler1
from lib.solver.build import make_lr_scheduler
from .inference import inference
from tqdm import tqdm
import os


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # print(all_losses)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    meters,
    arguments,
    param_wrapper=None,
    scehdulerorigin=None,
    cfg=None,
    train_name=None
):

    recon_dir = os.path.join(train_name, "recon")
    # if os.path.exists(recon_dir)==False:
    #     # os.mkdir(recon_dir)

    quant_dir = os.path.join(train_name, "quantmap")
    # if os.path.exists(quant_dir)==False:
    #     os.mkdir(quant_dir)

    config = cfg
    logger = logging.getLogger("PersonSearch.trainer")
    logger.info("Start training")

    max_epoch = arguments["max_epoch"]
    epoch = arguments["epoch"]
    max_iter = max_epoch * len(data_loader)
    iteration = arguments["iteration"]
    device = arguments["device"]
    checkpoint_period = arguments["checkpoint_period"]
    periodic_checkpoint = arguments["periodic_checkpoint"]
    evaluate_period = arguments["evaluate_period"]
    steps_to_accumulate = arguments['steps_to_accumulate']
    save_last_checkpoint = arguments['save_last_checkpoint']

    if arguments["amp"]:
        global amp_support
        if amp_support:
            assert not isinstance(model, torch.nn.DataParallel), \
                    "We do not support mixed precision training with DataParallel currently"
            grad_scaler = GradScaler()
            arguments["grad_scale"] = grad_scaler
        else:
            logger.info("Please update the PyTorch version (>=1.6) to support mixed precision training")
            grad_scaler = None

    best_top1 = arguments.get('best_top1', 0.0)
    best_epoch = arguments.get('best_epoch', 0)
    start_training_time = time.time()
    end = time.time()
    # print("==========models===========")
    # print(models)
    # print("==========models===========")

    while epoch < max_epoch:
        # FIXME: sampler's random seed is set each epoch.


        data_loader.batch_sampler.sampler.set_epoch(epoch)
        epoch += 1
        model.train()
        arguments["epoch"] = epoch
        print("length data loader", len((data_loader)))

        txt_cls     = [1, 1]  # acc_num, all_num
        quant_index = [0, 0]

        for step, (images, captions, _) in tqdm(enumerate(data_loader), ncols=100):
            # break
            # if step==10:
            #  break

            metric_dict = {}
            data_time = time.time() - end
            inner_iter = step
            iteration += 1
            arguments["iteration"] = iteration
            #
            images = images.to(device)
            captions = [caption.to(device) for caption in captions]
            batch_size = images.size()[0]

            #
            if arguments["amp"] and grad_scaler is not None:


                # logger.info("arguments['amp'] and grad_scaler is not None")

                with autocast():
                    # loss_dict, acc_dict = models(images, captions, img_mask=img_mask, epoch=epoch)
                    # optimizer.zero_grad()
                    #vae_embedding1 = model.vqvae.embedding
                    #vae_embedding2 = model.vqvae.embedding
                    loss_dict, acc_dict, other_dict = model(images, captions, epoch=epoch, recon_dir=recon_dir)


                    losses = sum(loss for loss in loss_dict.values())
                    losses /= steps_to_accumulate
                    if not np.isfinite(losses.detach().cpu().numpy()):
                        raise FloatingPointError(
                            f"Loss became infinite or NaN at epoch[{epoch}] iteration[{step}]!\n"
                            f"loss_dict = {loss_dict}"
                        )


                grad_scaler.scale(losses).backward()
                #print(vae_embedding2==vae_embedding1)
                #vae_embedding1=vae_embedding2
                if (step + 1) % steps_to_accumulate == 0:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()

                # # 用来计算txt分类准确率
                txt_cls[0] += other_dict['txt_cls_acc'][0]
                txt_cls[1] += other_dict['txt_cls_acc'][1]

                # 用来记录 quant_index 的分布。
                if type(quant_index[0]) == torch.Tensor:
                    quant_index[0] = torch.cat(
                        [quant_index[0], other_dict["quant_index"][0].reshape(batch_size, -1)])
                    quant_index[1] = torch.cat(
                        [quant_index[1], other_dict["quant_index"][1].reshape(batch_size, -1)])
                else:
                    quant_index[0] = other_dict["quant_index"][0]
                    quant_index[1] = other_dict["quant_index"][1]


            else:
                # visual feature: B*dim*H*W          [64, 2048, 24, 8]
                # textual feature: B*max_length*dim  [64, 64, 768]
                loss_dict, acc_dict = model(images, captions)
                losses = sum(loss for loss in loss_dict.values())
                losses /= steps_to_accumulate
                if not np.isfinite(losses.detach().cpu().numpy()):
                    raise FloatingPointError(
                        f"Loss became infinite or NaN at epoch[{epoch}] iteration[{step}]!\n"
                        f"loss_dict = {loss_dict}"
                    )

                losses.backward()
                if (step + 1) % steps_to_accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if isinstance(param_wrapper, ContiguousParams):
                param_wrapper.assert_buffer_is_valid()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            if acc_dict:
                acc_dict_reduced = reduce_loss_dict(acc_dict)
            synchronize()
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            metric_dict.update({'loss': losses_reduced})
            metric_dict.update(loss_dict_reduced)
            if acc_dict:
                metric_dict.update(acc_dict_reduced)

            batch_time = time.time() - end
            end = time.time()
            metric_dict.update({'time': batch_time, 'data': data_time, 'lr': optimizer.param_groups[0]["lr"]})

            meters.update(**metric_dict)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if inner_iter % arguments["print_iter"] == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{inner_iter}/{num_iter}]",
                            "{meters}",
                            "txt_cls_acc: {txt_cls_acc}%",
                            "max mem reserved: {memory:.0f}M"
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        inner_iter=inner_iter,
                        num_iter=len(data_loader),
                        meters=str(meters),
                        txt_cls_acc=round(float(txt_cls[0] / txt_cls[1]) * 100, 2),
                        memory=torch.cuda.max_memory_reserved() / 1024.0 / 1024.0,
                    )
                )

        scheduler.step()

        if epoch % evaluate_period == 0:


            top1 = inference(model, data_loader_val, dataset_name=arguments['dataset_val'], save_data=False, rerank=False)
            top1 = all_gather(top1)[0]
            torch.cuda.empty_cache()
            synchronize()
            meters.update(top1=top1.item())
            if top1.item() > best_top1:
                best_top1 = top1.item()
                best_epoch = epoch
                arguments['best_top1'] = best_top1
                arguments['best_epoch'] = best_epoch
                checkpointer.save("best", **arguments)
            logger.info("Total evaluation time: {}".format(str(datetime.timedelta(seconds=time.time() - end))))
            logger.info("Current Best models in epoch {} with top1 {}".format(best_epoch, best_top1))

        if periodic_checkpoint and epoch % checkpoint_period == 0:
            checkpointer.save("epoch_{:d}".format(epoch), **arguments)
        if save_last_checkpoint:
            checkpointer.save("last_checkpoint", **arguments)


        end = time.time()

    logger.info(
        "Best models in epoch {} with top1 {}".format(best_epoch, best_top1)
    )

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
