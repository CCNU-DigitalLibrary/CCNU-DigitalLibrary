import argparse
import os
import random
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from fvcore.common.checkpoint import Checkpointer

from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.trainer import do_train



# from lib.models.model import build_model

from lib.models.modelcodebook import build_modelcodebook

from lib.solver import make_lr_scheduler, make_optimizer
from lib.utils.collect_env import collect_env_info
from lib.utils.comm import get_rank, synchronize, is_main_process
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger
from lib.utils.metric_logger import MetricLogger, TensorboardLogger
from lib.utils.misc import cp_projects

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





def set_random_seed(seed=0, deterministic=False, cudnn_benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark


def set_freeze_iters(cfg, data_loader):
    # multiple freeze epochs by iters_per_epoch
    cfg.defrost()
    cfg.SOLVER.FREEZE_EPOCHS *= len(data_loader)
    cfg.freeze()
    return cfg


def train(cfg, output_dir, args):
    local_rank = args.local_rank
    distributed = args.distributed
    resume_from = args.resume_from
    use_tensorboard = args.use_tensorboard

    # build up train loader
    data_loader_train = make_data_loader(
        cfg,
        is_train=True,
    )
    # build up val loader
    data_loader_val = make_data_loader(
        cfg,
        is_train=True,
        is_val=True,
    )
    # set freeze iterations
    cfg = set_freeze_iters(cfg, data_loader_train)


    # # build models
    models = build_modelcodebook(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    models.to(device)
    # #
    # print(models)
    # # #
    # build optimizer
    optimizer, param_wrapper = make_optimizer(cfg, models, contiguous=cfg.SOLVER.CONTIGUOUS_PARAMS)
    # build lr_scheduler
    scheduler = make_lr_scheduler(cfg, optimizer)
    #
    if distributed:
        models = torch.nn.parallel.DistributedDataParallel(
            models,
            device_ids=[local_rank],
            output_device=local_rank,
            # # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    #
    arguments = {}
    arguments["iteration"] = 0
    arguments["epoch"] = 0

    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(models, output_dir, save_to_disk=save_to_disk, optimizer=optimizer, scheduler=scheduler)
    if cfg.MODEL.WEIGHT != "imagenet":
        if os.path.isfile(cfg.MODEL.WEIGHT):
            checkpointer.load(cfg.MODEL.WEIGHT, checkpointables=cfg.SOLVER.CHECKPOINT_TABLES)
        else:
            raise IOError("{} is not a checkpoint file".format(cfg.MODEL.WEIGHT))
    #
    if args.resume_from_suspension:
        # resume last checkpoint from suspension
        try:
            if os.path.isfile(os.path.join(output_dir, "last_checkpoint")):
                extra_checkpoint_data = checkpointer.resume_or_load(output_dir, resume=True)
                arguments.update(extra_checkpoint_data)
                optimizer.load_state_dict(arguments.get("optimizer", optimizer.state_dict()))
                scheduler.load_state_dict(arguments.get("scheduler", scheduler.state_dict()))
                print('resume from last_checkpoint')
            else:
                print(f'no last_checkpoint.pth found in {output_dir}')
        except:
            print('resume from last_checkpoint error!!! this might caused by saving the broken last_checkpoint.pth.')
            print('try to resume from best.pth.')
            if os.path.isfile(os.path.join(output_dir, "best.pth")):
                extra_checkpoint_data = checkpointer.resume_or_load(os.path.join(output_dir, "best.pth"))
                arguments.update(extra_checkpoint_data)
                optimizer.load_state_dict(arguments.get("optimizer", optimizer.state_dict()))
                scheduler.load_state_dict(arguments.get("scheduler", scheduler.state_dict()))
            else:
                print(f'no best.pth found in {output_dir}')

    if resume_from:
        if os.path.isfile(resume_from):
            extra_checkpoint_data = checkpointer.resume_or_load(resume_from)
            arguments.update(extra_checkpoint_data)
            optimizer.load_state_dict(arguments.get("optimizer", optimizer.state_dict()))
            scheduler.load_state_dict(arguments.get("scheduler", scheduler.state_dict()))
        else:
            raise IOError("{} is not a checkpoint file".format(resume_from))

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(output_dir, "tensorboard"),
            start_iter=arguments["iteration"],
            delimiter="  ",
        )
    else:
        meters = MetricLogger(delimiter="  ")
    #
    arguments['checkpoint_period'] = cfg.SOLVER.CHECKPOINT_PERIOD
    arguments['evaluate_period'] = cfg.SOLVER.EVALUATE_PERIOD
    arguments["max_epoch"] = cfg.SOLVER.NUM_EPOCHS
    arguments["distributed"] = distributed
    arguments['device'] = cfg.MODEL.DEVICE
    arguments['dataset_train'] = cfg.DATASETS.TRAIN
    arguments['dataset_val'] = cfg.DATASETS.VAL
    arguments['amp'] = cfg.SOLVER.AMP
    arguments['print_iter'] = cfg.SOLVER.PRINT_ITER
    arguments['periodic_checkpoint'] = cfg.SOLVER.PERIODIC_CHECKPOINT
    arguments['steps_to_accumulate'] = cfg.SOLVER.STEPS_TO_ACCUMULATE
    arguments['save_last_checkpoint'] = cfg.SOLVER.SAVE_LAST_CHECKPOINT

    train_name = cfg.MODE.TRAIN_NAME

    do_train(
        models,
        data_loader_train,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        meters,
        arguments,
        param_wrapper,
        trainname=train_name
    )


def main():
    start_time = time.monotonic()

    parser = argparse.ArgumentParser(description="PyTorch Person Search Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--resume-from",
        help="the checkpoint file to resume from",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
    )
    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboard logger (Requires tensorboard installed)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="subfolder name for logging and checkpoint"
    )
    # arguments for hfai
    parser.add_argument(
        "--without-timestamp",
        action="store_true",
        help="append time stamp after the name of output dir."
    )
    parser.add_argument(
        "--resume-from-suspension",
        action="store_true",
        help="resume last_checkpoint from suspension"
    )

    args = parser.parse_args()


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.distributed = num_gpus > 1
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0




    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_random_seed(cfg.EXP.SEED, cfg.EXP.DETERMINISTIC, cfg.EXP.CUDNN_BENCHMARK)

    # print(cfg.MODE.MIM)
    # print(cfg.MODE.MLM)
    # print(cfg.MODE.CODEBOOK)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    #
    #
    output_dir = 'train_' + time.strftime("%Y-%m-%dT%H-%M-%S") if not args.without_timestamp else 'train'
    output_dir = os.path.join("./output", args.config_file[8:-5], args.output_dir, output_dir)
    if is_main_process():
        makedir(output_dir)
    # #
    logger = setup_logger("PersonSearch", output_dir, get_rank())
    logger.info("Environment info:\n{}".format(collect_env_info()))
    logger.info("Rank of current process: {}. Using {} GPUs".format(get_rank(), num_gpus))
    logger.info("Command line arguments: {}".format(str(args)))
    #
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    if is_main_process():
        shutil.copy(args.config_file, os.path.join(output_dir, os.path.basename(args.config_file)))
        if cfg.EXP.CP_PROJECT:

            logger.info("Do not save full project saved to {}".format(output_dir))
            # logger.info("Full project saved to {}".format(output_dir))
            # cp_projects(output_dir)
    # #
    train(
        cfg,
        output_dir,
        args,
    )
    #
    # # print time
    end_time = time.monotonic()
    logger.info(f"Total running time: {timedelta(seconds=end_time - start_time)}")


if __name__ == "__main__":
    main()
