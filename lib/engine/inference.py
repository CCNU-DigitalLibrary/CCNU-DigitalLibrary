import datetime
import logging
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from copy import deepcopy

from lib.data.metrics import evaluation_common, evaluation_cross, evaluation_common_hash
from lib.utils.comm import all_gather, is_main_process, synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = defaultdict(list)
    for batch in tqdm(data_loader):
        # print(batch)
        images, captions, image_ids= batch
        images = images.to(device)
        captions = [caption.to(device) for caption in captions]
        with torch.no_grad():
            output = model(images, captions, mode="val") # visual embedding and textual embedding

        for result in output:
            if isinstance(result, list):  # multiple features, stored in List
                for idx, img_id in enumerate(image_ids):
                    pred_list = []
                    for res in result:
                        pred_list.append(deepcopy(res[idx].to(torch.device('cpu'))))  # save memory
                    results_dict[img_id].append(pred_list)
            else:
                for img_id, pred in zip(image_ids, result):
                    # print(img_id, pred, pred.size())
                    results_dict[img_id].append(deepcopy(pred.to(torch.device('cpu'))))  # save memory
        del images
        del captions
        del output
    # for k, y in results_dict.items():
    #     print(k, len(y), y[0].size(), y[1].size())
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("PersonSearch.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    return predictions


def inference(
    model,
    data_loader,
    dataset_name="cuhkpedes-test",
    device="cuda",
    output_folder="",
    save_data=True,
    rerank=True,
    sum_sim=True
):
    logger = logging.getLogger("PersonSearch.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset))
    )

    predictions = None
    if not os.path.exists(os.path.join(output_folder, "inference_data.npz")):
        # convert to a torch.device for efficiency
        device = torch.device(device)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        start_time = time.time()

        predictions = compute_on_dataset(model, data_loader, device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

        if not is_main_process():
            return
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        embed_model = model.module.embed_model
    else:
        embed_model = model.embed_model

    if not hasattr(embed_model, "inference_mode") or embed_model.inference_mode == "common":
        return evaluation_common(
        # return evaluation_common_hash(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            save_data=save_data,
            rerank=rerank,
            sum_sim=sum_sim,
            topk=[1, 5, 10],
        )

    if embed_model.inference_mode == "cross":
        assert hasattr(embed_model, "get_similarity")
        sim_calculator = embed_model.get_similarity
        return evaluation_cross(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            save_data=save_data,
            sim_calculator=sim_calculator,
            topk=[1, 5, 10],
        )
