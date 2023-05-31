import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.logger import table_log


def rank(similarity, q_pids, g_pids, topk=[1, 5, 10], get_mAP=True, ignore_self=False):
    max_rank = max(topk) if not ignore_self else max(topk) + 1
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    if ignore_self:
        # remove the self similarity
        matches = matches[:, 1:]

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / (num_rel + 1e-8) # q
    mAP = AP.mean() * 100
    #     np.save("cmc.npy", matches)
    #     np.save("similarity.npy", similarity)
    #     np.save("q_pids.npy", q_pids)
    #     np.save("g_pids.npy", g_pids)
    #     np.save("indices.npy", indices)
    return all_cmc, mAP, indices


def jaccard(a_list, b_list):
    return float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list)))


def jaccard_mat(row_nn, col_nn):
    jaccard_sim = np.zeros((row_nn.shape[0], col_nn.shape[0]))
    # FIXME: need optimization
    for i in range(row_nn.shape[0]):
        for j in range(col_nn.shape[0]):
            jaccard_sim[i, j] = jaccard(row_nn[i], col_nn[j])
    return torch.from_numpy(jaccard_sim)


def k_reciprocal(q_feats, g_feats, neighbor_num=5, alpha=0.05):
    qg_sim = torch.matmul(q_feats, g_feats.t())  # q * g
    gg_sim = torch.matmul(g_feats, g_feats.t())  # g * g

    qg_indices = torch.argsort(qg_sim, dim=1, descending=True)
    gg_indices = torch.argsort(gg_sim, dim=1, descending=True)

    qg_nn = qg_indices[:, :neighbor_num]  # q * n
    gg_nn = gg_indices[:, :neighbor_num]  # g * n

    jaccard_sim = jaccard_mat(qg_nn.cpu().numpy(), gg_nn.cpu().numpy())  # q * g
    jaccard_sim = jaccard_sim.to(qg_sim.device)
    return alpha * jaccard_sim  # q * g


def get_unique(image_ids):
    keep_idx = {}
    for idx, image_id in enumerate(image_ids):
        if image_id not in keep_idx.keys():
            keep_idx[image_id] = idx
    return torch.tensor(list(keep_idx.values()))


def evaluation_common(
    dataset,
    predictions,
    output_folder,
    topk,
    save_data=True,
    rerank=True,
    sum_sim=True
):
    logger = logging.getLogger("PersonSearch.inference")
    data_dir = os.path.join(output_folder, "inference_data.npz")

    if predictions is None:
        inference_data = np.load(data_dir)
        logger.info("Load inference data from {}".format(data_dir))
        image_pid = torch.tensor(inference_data["image_pid"])
        text_pid = torch.tensor(inference_data["text_pid"])
        similarity = torch.tensor(inference_data["similarity"])
        similarity_t2t = torch.tensor(inference_data["similarity_t2t"])
        similarity_i2i = torch.tensor(inference_data["similarity_i2i"])
        if rerank:
            rvn_mat = torch.tensor(inference_data["rvn_mat"])
            rtn_mat = torch.tensor(inference_data["rtn_mat"])
    else:
        image_ids, pids = [], []
        image_global, text_global = [], []

        # FIXME: need optimization
        for idx, prediction in predictions.items():
            image_id, pid = dataset.get_id_info(idx)
            image_ids.append(image_id)
            pids.append(pid)
            image_global.append(prediction[0])
            text_global.append(prediction[1])

        if isinstance(image_global[0], list) and isinstance(text_global[0], list):
            # multiple features stored in list
            image_pid = torch.tensor(pids)
            text_pid = torch.tensor(pids)
            num_feat = len(image_global[0])
            assert num_feat == len(text_global[0]), f"num of image feat({num_feat}) != num of text feat({len(text_global[0])})"
            
            keep_idx = get_unique(image_ids)
            image_global = [image_global[idx] for idx in keep_idx.tolist()] 
            image_pid = image_pid[keep_idx]

            similarity = []
            similarity_t2t = []
            similarity_i2i = []
            rtn_mat = []
            rvn_mat = []
            for feat_idx in range(num_feat):
                image_i = torch.stack([image_global[i][feat_idx] for i in range(len(image_global))], dim=0)
                text_i = torch.stack([text_global[i][feat_idx] for i in range(len(text_global))], dim=0)

                image_i = F.normalize(image_i, p=2, dim=1)
                text_i = F.normalize(text_i, p=2, dim=1)

                similarity_i = torch.matmul(text_i, image_i.t())
                similarity_i_t2t = torch.matmul(text_i, text_i.t())
                similarity_i_i2i = torch.matmul(image_i, image_i.t())

                similarity.append(similarity_i)
                similarity_t2t.append(similarity_i_t2t)
                similarity_i2i.append(similarity_i_i2i)

                if rerank:
                    rtn_mat_i = k_reciprocal(image_i, text_i)
                    rvn_mat_i = k_reciprocal(text_i, image_i)

                    rtn_mat.append(rtn_mat_i)
                    rvn_mat.append(rvn_mat_i)

            if sum_sim:
                # we simply use mean of the similarity matrix as the global similarity
                similarity = sum(similarity)
                similarity_t2t = sum(similarity_t2t)
                similarity_i2i = sum(similarity_i2i)
                if rerank:
                    rtn_mat = sum(rtn_mat)
                    rvn_mat = sum(rvn_mat)
        else:
            # single global feature
            image_pid = torch.tensor(pids)
            text_pid = torch.tensor(pids)
            image_global = torch.stack(image_global, dim=0)
            text_global = torch.stack(text_global, dim=0)

            keep_idx = get_unique(image_ids)
            image_global = image_global[keep_idx]
            image_pid = image_pid[keep_idx]

            image_global = F.normalize(image_global, p=2, dim=1)
            text_global = F.normalize(text_global, p=2, dim=1)

            similarity = torch.matmul(text_global, image_global.t())
            similarity_t2t = torch.matmul(text_global, text_global.t())
            similarity_i2i = torch.matmul(image_global, image_global.t())

            if rerank:
                rtn_mat = k_reciprocal(image_global, text_global)
                rvn_mat = k_reciprocal(text_global, image_global)

    if save_data:
        if not rerank:
            np.savez(
                data_dir,
                image_pid=image_pid.cpu().numpy(),
                text_pid=text_pid.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                similarity_t=similarity_t2t.cpu().numpy(),
                similarity_i=similarity_i2i.cpu().numpy()
            )
        else:
            np.savez(
                data_dir,
                image_pid=image_pid.cpu().numpy(),
                text_pid=text_pid.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                similarity_t=similarity_t2t.cpu().numpy(),
                similarity_i=similarity_i2i.cpu().numpy(),
                rvn_mat=rvn_mat.cpu().numpy(),
                rtn_mat=rtn_mat.cpu().numpy(),
            )

    topk = torch.tensor(topk)

    if rerank:
        i2t_cmc, i2t_mAP, _ = rank(similarity.t(), image_pid, text_pid, topk, get_mAP=True)
        t2i_cmc, t2i_mAP, _ = rank(similarity, text_pid, image_pid, topk, get_mAP=True)
        re_i2t_cmc, re_i2t_mAP, _ = rank(rtn_mat + similarity.t(), image_pid, text_pid, topk, get_mAP=True)
        re_t2i_cmc, re_t2i_mAP, _ = rank(rvn_mat + similarity, text_pid, image_pid, topk, get_mAP=True)
        t2t_cmc, t2t_mAP, _ = rank(similarity_t2t, text_pid, text_pid, topk, get_mAP=True, ignore_self=True)
        i2i_cmc, i2i_mAP, _ = rank(similarity_i2i, image_pid, image_pid, topk, get_mAP=True, ignore_self=True)
        cmc_results = torch.stack([topk, t2i_cmc, re_t2i_cmc, i2t_cmc, re_i2t_cmc, t2t_cmc, i2i_cmc])
        mAP_results = torch.stack([
            torch.zeros_like(t2i_mAP), t2i_mAP, re_t2i_mAP, i2t_mAP, re_i2t_mAP, t2t_mAP, i2i_mAP
        ]).unsqueeze(-1)
        results = torch.cat([cmc_results, mAP_results], dim=1)
        results = results.t().cpu().numpy().tolist()
        results[-1][0] = "mAP"
        logger.info(
            "\n"
            + table_log(results, headers=["topk", "t2i", "re-t2i", "i2t", "re-i2t", "t2t", "i2i"])
        )
    else:
        if not isinstance(similarity, list):
            similarity, similarity_t2t, similarity_i2i = [similarity], [similarity_t2t], [similarity_i2i]
        for i in range(len(similarity)):
            similarity_i, similarity_t2t_i, similarity_i2i_i = similarity[i], similarity_t2t[i], similarity_i2i[i]

            t2i_cmc, t2i_mAP, _ = rank(similarity_i, text_pid, image_pid, topk, get_mAP=True)
            i2t_cmc, i2t_mAP, _ = rank(similarity_i.t(), image_pid, text_pid, topk, get_mAP=True)
            t2t_cmc, t2t_mAP, _ = rank(similarity_t2t_i, text_pid, text_pid, topk, get_mAP=True, ignore_self=True)
            i2i_cmc, i2i_mAP, _ = rank(similarity_i2i_i, image_pid, image_pid, topk, get_mAP=True, ignore_self=True)
            cmc_results = torch.stack((topk, t2i_cmc, i2t_cmc, t2t_cmc, i2i_cmc))
            mAP_results = torch.stack(
                [torch.zeros_like(t2i_mAP), t2i_mAP, i2t_mAP, t2t_mAP, i2i_mAP]
            ).unsqueeze(-1)
            results = torch.cat([cmc_results, mAP_results], dim=1)
            results = results.t().cpu().numpy().tolist()
            results[-1][0] = "mAP"
            for i, k in enumerate(topk.cpu().numpy().tolist()):
                results[i][0] = k
            logger.info("\n" + table_log(results, headers=["topk", "t2i", "i2t", "t2t", "i2i"]))
    return t2i_cmc[0]



def evaluation_common_hash(
        dataset,
        predictions,
        output_folder,
        topk,
        save_data=True,
        rerank=True,
        sum_sim=True
):
    logger = logging.getLogger("PersonSearch.inference")
    data_dir = os.path.join(output_folder, "inference_data.npz")

    if predictions is None:
        inference_data = np.load(data_dir)
        logger.info("Load inference data from {}".format(data_dir))
        image_pid = torch.tensor(inference_data["image_pid"])
        text_pid = torch.tensor(inference_data["text_pid"])
        similarity = torch.tensor(inference_data["similarity"])
        similarity_t2t = torch.tensor(inference_data["similarity_t2t"])
        similarity_i2i = torch.tensor(inference_data["similarity_i2i"])
        if rerank:
            rvn_mat = torch.tensor(inference_data["rvn_mat"])
            rtn_mat = torch.tensor(inference_data["rtn_mat"])
    else:
        image_ids, pids = [], []
        image_global, text_global = [], []

        # FIXME: need optimization
        for idx, prediction in predictions.items():
            image_id, pid = dataset.get_id_info(idx)
            image_ids.append(image_id)
            pids.append(pid)
            image_global.append(prediction[0])
            text_global.append(prediction[1])

        if isinstance(image_global[0], list) and isinstance(text_global[0], list):
            # multiple features stored in list
            image_pid = torch.tensor(pids)
            text_pid = torch.tensor(pids)
            num_feat = len(image_global[0])
            assert num_feat == len(
                text_global[0]), f"num of image feat({num_feat}) != num of text feat({len(text_global[0])})"

            keep_idx = get_unique(image_ids)
            image_global = [image_global[idx] for idx in keep_idx.tolist()]
            image_pid = image_pid[keep_idx]


            similarity = []
            similarity_t2t = []
            similarity_i2i = []
            rtn_mat = []
            rvn_mat = []
            for feat_idx in range(num_feat):
                image_i = torch.stack([image_global[i][feat_idx] for i in range(len(image_global))], dim=0)
                text_i = torch.stack([text_global[i][feat_idx] for i in range(len(text_global))], dim=0)

                image_i = F.normalize(image_i, p=2, dim=1)
                text_i = F.normalize(text_i, p=2, dim=1)

                similarity_i = torch.matmul(text_i, image_i.t())
                similarity_i_t2t = torch.matmul(text_i, text_i.t())
                similarity_i_i2i = torch.matmul(image_i, image_i.t())

                similarity.append(similarity_i)
                similarity_t2t.append(similarity_i_t2t)
                similarity_i2i.append(similarity_i_i2i)

                if rerank:
                    rtn_mat_i = k_reciprocal(image_i, text_i)
                    rvn_mat_i = k_reciprocal(text_i, image_i)

                    rtn_mat.append(rtn_mat_i)
                    rvn_mat.append(rvn_mat_i)

            if sum_sim:
                # we simply use mean of the similarity matrix as the global similarity
                similarity = sum(similarity)
                similarity_t2t = sum(similarity_t2t)
                similarity_i2i = sum(similarity_i2i)
                if rerank:
                    rtn_mat = sum(rtn_mat)
                    rvn_mat = sum(rvn_mat)
        else:
            # single global feature
            image_pid = torch.tensor(pids)
            text_pid = torch.tensor(pids)
            image_global = torch.stack(image_global, dim=0)
            text_global = torch.stack(text_global, dim=0)

            keep_idx = get_unique(image_ids)
            image_global = image_global[keep_idx]
            image_pid = image_pid[keep_idx]

            image_global = F.normalize(image_global, p=2, dim=1)
            text_global = F.normalize(text_global, p=2, dim=1)

            similarity_hash = HammingD(text_global, image_global)
            # print(similarity_hash, similarity_hash.shape)
            similarity = torch.matmul(text_global, image_global.t())
            similarity_t2t = torch.matmul(text_global, text_global.t())
            similarity_i2i = torch.matmul(image_global, image_global.t())

            if rerank:
                rtn_mat = k_reciprocal(image_global, text_global)
                rvn_mat = k_reciprocal(text_global, image_global)

    if save_data:
        if not rerank:
            np.savez(
                data_dir,
                image_pid=image_pid.cpu().numpy(),
                text_pid=text_pid.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                similarity_t=similarity_t2t.cpu().numpy(),
                similarity_i=similarity_i2i.cpu().numpy()
            )
        else:
            np.savez(
                data_dir,
                image_pid=image_pid.cpu().numpy(),
                text_pid=text_pid.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                similarity_t=similarity_t2t.cpu().numpy(),
                similarity_i=similarity_i2i.cpu().numpy(),
                rvn_mat=rvn_mat.cpu().numpy(),
                rtn_mat=rtn_mat.cpu().numpy(),
            )

    topk = torch.tensor(topk)

    if rerank:
        i2t_cmc, i2t_mAP, _ = rank(similarity.t(), image_pid, text_pid, topk, get_mAP=True)
        t2i_cmc, t2i_mAP, _ = rank(similarity, text_pid, image_pid, topk, get_mAP=True)
        re_i2t_cmc, re_i2t_mAP, _ = rank(rtn_mat + similarity.t(), image_pid, text_pid, topk, get_mAP=True)
        re_t2i_cmc, re_t2i_mAP, _ = rank(rvn_mat + similarity, text_pid, image_pid, topk, get_mAP=True)
        t2t_cmc, t2t_mAP, _ = rank(similarity_t2t, text_pid, text_pid, topk, get_mAP=True, ignore_self=True)
        i2i_cmc, i2i_mAP, _ = rank(similarity_i2i, image_pid, image_pid, topk, get_mAP=True, ignore_self=True)
        cmc_results = torch.stack([topk, t2i_cmc, re_t2i_cmc, i2t_cmc, re_i2t_cmc, t2t_cmc, i2i_cmc])
        mAP_results = torch.stack([
            torch.zeros_like(t2i_mAP), t2i_mAP, re_t2i_mAP, i2t_mAP, re_i2t_mAP, t2t_mAP, i2i_mAP
        ]).unsqueeze(-1)
        results = torch.cat([cmc_results, mAP_results], dim=1)
        results = results.t().cpu().numpy().tolist()
        results[-1][0] = "mAP"
        logger.info(
            "\n"
            + table_log(results, headers=["topk", "t2i", "re-t2i", "i2t", "re-i2t", "t2t", "i2i"])
        )
    else:
        if not isinstance(similarity, list):
            similarity, similarity_t2t, similarity_i2i = [similarity], [similarity_t2t], [similarity_i2i]
            similarity_hash = torch.from_numpy(similarity_hash)
            similarity_hash = [similarity_hash]
        for i in range(len(similarity)):
            similarity_i, similarity_t2t_i, similarity_i2i_i = similarity[i], similarity_t2t[i], similarity_i2i[i]

            t2i_cmc, t2i_mAP, _ = rank(similarity_hash[i], text_pid, image_pid, topk, get_mAP=True)
            i2t_cmc, i2t_mAP, _ = rank(similarity_hash[i].t(), image_pid, text_pid, topk, get_mAP=True)

            # t2i_cmc, t2i_mAP, _ = rank(similarity_i, text_pid, image_pid, topk, get_mAP=True)
            # i2t_cmc, i2t_mAP, _ = rank(similarity_i.t(), image_pid, text_pid, topk, get_mAP=True)
            t2t_cmc, t2t_mAP, _ = rank(similarity_t2t_i, text_pid, text_pid, topk, get_mAP=True, ignore_self=True)
            i2i_cmc, i2i_mAP, _ = rank(similarity_i2i_i, image_pid, image_pid, topk, get_mAP=True, ignore_self=True)
            cmc_results = torch.stack((topk, t2i_cmc, i2t_cmc, t2t_cmc, i2i_cmc))
            mAP_results = torch.stack(
                [torch.zeros_like(t2i_mAP), t2i_mAP, i2t_mAP, t2t_mAP, i2i_mAP]
            ).unsqueeze(-1)
            results = torch.cat([cmc_results, mAP_results], dim=1)
            results = results.t().cpu().numpy().tolist()
            results[-1][0] = "mAP"
            for i, k in enumerate(topk.cpu().numpy().tolist()):
                results[i][0] = k
            logger.info("\n" + table_log(results, headers=["topk", "t2i", "i2t", "t2t", "i2i"]))
    return t2i_cmc[0]

def evaluation_cross(
    dataset,
    predictions,
    output_folder,
    topk,
    save_data=True,
    sim_calculator=None,
):
    logger = logging.getLogger("PersonSearch.inference")
    data_dir = os.path.join(output_folder, "inference_data.npz")

    if predictions is None:
        inference_data = np.load(data_dir)
        logger.info("Load inference data from {}".format(data_dir))
        image_pid = torch.tensor(inference_data["image_pid"])
        text_pid = torch.tensor(inference_data["text_pid"])
        similarity = torch.tensor(inference_data["similarity"])
    else:
        image_ids, pids = [], []
        patch_embed, word_embed, key_padding_mask = [], [], []

        # FIXME: need optimization
        for idx, prediction in predictions.items():
            image_id, pid = dataset.get_id_info(idx)
            image_ids.append(image_id)
            pids.append(pid)
            patch_embed.append(prediction[0])
            word_embed.append(prediction[1])
            key_padding_mask.append(prediction[2])

        image_pid = torch.tensor(pids)
        text_pid = torch.tensor(pids)
        patch_embed = torch.stack(patch_embed, dim=0)
        word_embed = torch.stack(word_embed, dim=0)
        key_padding_mask = torch.stack(key_padding_mask, dim=0)

        keep_id = get_unique(image_ids)
        patch_embed = patch_embed[keep_id]
        image_pid = image_pid[keep_id]

        with torch.no_grad():
            similarity = sim_calculator(
                patch_embed, word_embed, key_padding_mask, chunk_size=1024
            )

        if save_data:
            np.savez(
                data_dir,
                image_pid=image_pid.cpu().numpy(),
                text_pid=text_pid.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
            )

    topk = torch.tensor(topk)

    t2i_cmc, _ = rank(similarity, text_pid, image_pid, topk, get_mAP=False)
    i2t_cmc, _ = rank(similarity.t(), image_pid, text_pid, topk, get_mAP=False)
    results = torch.stack((topk, t2i_cmc, i2t_cmc)).t().cpu().numpy()
    logger.info("\n" + table_log(results, headers=["topk", "t2i", "i2t"]))
    return t2i_cmc[0]

def HammingD(q_feature,  d_feature, R_h=None):
    '''
    HammingD: compare hamming distance for query and database
              modified for our model
    args:
        f1: dense feature of the query
        binary_f: binary hashing code of the database items
        R_h: only output top-R_h most similar items from the database
    '''
    # print("hammdingD")
    q_feature = q_feature.cpu().detach().numpy()
    d_feature = d_feature.cpu().detach().numpy()

    # q_b = 2*np.int8(q_feature>=0)-1
    # d_b = 2*np.int8(d_feature>=0)-1
    q_b = np.sign(q_feature)
    d_b = np.sign(d_feature)


    # print(q_feature)
    # print(q_b, sum(q_b[0]))



    #print('b',B)
    dis = np.dot(q_b,np.transpose(d_b))
    ids = np.argsort(-dis, 1)
    # APx = []
    # ID = ids[:,0:R_h]
    # print(dis)
    return dis