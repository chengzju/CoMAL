import numpy as np

from config import *
from samplers import SubsetSequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from networks import *
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from apex import amp
from utils import *
from kcenterGreedy import *
import pdb
import os
from scipy import stats


def query_samples(args, cycle, models, dataset, subset, device, label_num,
                  label_cardinality=None, labeled_subset = None, record_path=None):
    for key in models.keys():
        models[key].eval()
    all_preds, all_indices = [], []
    all_loss_weight = []
    all_back_preds = []
    all_back_dists = []
    all_gt_labels = []
    all_dual_preds = []
    all_sub_reps = []
    if record_path is not None:
        this_record_path = os.path.join(record_path, 'cycle_{}_cnt_0'.format(cycle))
    if args.train_vae_module:
        all_labed_sub_rebs = []
        all_labed_dists = []
        all_labed_gt_labels = []
        labeled_dataloader = DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        num_workers=NUM_WORKER,
                                        pin_memory=True,
                                        sampler=SubsetRandomSampler(labeled_subset)
                                        )

        for text_inputs, gt_label, idx, _, _ in tqdm(labeled_dataloader):
            input_ids, input_type_ids, input_mask = [x.to(device) for x in text_inputs]
            all_labed_gt_labels.extend(gt_label)
            with torch.no_grad():
                _, _, encoder_cls = models['backbone']([input_ids, input_type_ids, input_mask, None])
                if 'vae_module' in models.keys():
                    _, sub_rep, _, vae_cl_dist = models['vae_module'](encoder_cls)
                    sub_rep = sub_rep.cpu().data # N, L, d
                    vae_cl_dist = vae_cl_dist.cpu().data
                    all_labed_dists.extend(vae_cl_dist)
                    all_labed_sub_rebs.extend(F.normalize(sub_rep, dim=-1))
        all_labed_gt_labels = torch.stack(all_labed_gt_labels) # N, L
        all_labed_sub_rebs = torch.stack(all_labed_sub_rebs) # N, L, d
        all_labed_dists = torch.stack(all_labed_dists) # N, L+1
        if args.save_sub_tensor:
            labed_gt_label_path = os.path.join(this_record_path, 'labed_labels.pt')
            labed_sub_path = os.path.join(this_record_path, 'labed_sub.pt')
            torch.save(all_labed_gt_labels, labed_gt_label_path)
            torch.save(all_labed_sub_rebs, labed_sub_path)
        if args.cl_neg_mode == 1:
            labed_self_proto_dists = all_labed_dists[:,:, :-1]  # N, L, L
            labed_self_proto_dists = labed_self_proto_dists * torch.eye(label_num)  # N, L, L
            labed_self_proto_dists = torch.sum(labed_self_proto_dists, dim=-1) # N, L
            labed_pos_self_dists = labed_self_proto_dists * all_labed_gt_labels # N, L
        elif args.cl_neg_mode == 2:
            labed_pos_self_dists, labed_neg_self_dists = all_labed_dists.chunk(2, dim=-1)
            labed_pos_self_dists = labed_pos_self_dists * all_labed_gt_labels  # N, L
        labed_pos_self_dists_max = torch.max(labed_pos_self_dists, dim=0)[0]
        labed_pos_self_dists_min = torch.min(labed_pos_self_dists + (1 - all_labed_gt_labels) * 2,dim=0)[0]
        labed_pos_self_dists_mean = (labed_pos_self_dists_max + labed_pos_self_dists_min) / 2

    unlabeled_dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      num_workers=NUM_WORKER,
                                      pin_memory=True,
                                      sampler=SubsetRandomSampler(subset)
                                      )
    for text_inputs, labels, idx, loss_weight, _ in tqdm(unlabeled_dataloader):
        input_ids, input_type_ids, input_mask = [x.to(device) for x in text_inputs]
        with torch.no_grad():
            preds, atten_out, encoder_cls = models['backbone']([input_ids, input_type_ids, input_mask, None])
            preds = preds.cpu().data
            all_back_preds.extend(preds)
            if 'vae_module' in models.keys():
                recon_x, latent_rep, recon_x_clf, vae_cl_dist = models['vae_module'](encoder_cls)
                latent_rep = latent_rep.cpu().data
                vae_cl_dist = vae_cl_dist.cpu().data
                recon_x_clf = recon_x_clf.cpu().data
                all_dual_preds.extend(recon_x_clf)
                all_back_dists.extend(vae_cl_dist)
                all_sub_reps.extend(F.normalize(latent_rep, dim=-1))
            else:
                all_dual_preds.extend(preds)
        all_indices.extend(idx)
        all_loss_weight.extend(loss_weight)
        all_gt_labels.extend(labels)
    all_indices = torch.stack(all_indices)
    all_loss_weight = torch.stack(all_loss_weight)
    all_gt_labels = torch.stack(all_gt_labels)
    if 'vae_module' in models.keys():
        all_sub_reps = torch.stack(all_sub_reps)
        if args.save_sub_tensor:
            unlabed_gt_label_path = os.path.join(this_record_path, 'unlabed_labels.pt')
            unlabed_sub_path = os.path.join(this_record_path, 'unlabed_sub.pt')
            torch.save(all_gt_labels, unlabed_gt_label_path)
            torch.save(all_sub_reps, unlabed_sub_path)

        back_dists_mat = torch.stack(all_back_dists)
    all_back_preds = torch.stack(all_back_preds)
    back_preds_mat = torch.zeros_like(all_back_preds) + all_back_preds
    back_preds_mat = torch.sigmoid(back_preds_mat)
    unlab_mask_mat = torch.ones_like(all_loss_weight) - all_loss_weight
    all_loss_weight = all_loss_weight.view(-1)
    part_unlabeled, part_labeled = pred_part_stat(all_loss_weight, thresold=0.5)  # no rank
    label_wise_part_unlabeled = list(set(part_unlabeled // label_num))
    label_wise_part_unlabeled = torch.tensor(label_wise_part_unlabeled)
    back_pred_pos_mask = ((back_preds_mat >= 0.5) * unlab_mask_mat).to(torch.float32)
    if args.train_vae_module:
        if args.cl_neg_mode == 2:
            sub_pos_proto_dists, sub_neg_proto_dists = back_dists_mat.chunk(2, dim=-1) # N, L
        elif args.cl_neg_mode == 0 or args.cl_neg_mode == 1:
            if args.cl_neg_mode == 0:
                sub_pos_proto_dists = back_dists_mat # N, L, L
            elif args.cl_neg_mode == 1:
                sub_pos_proto_dists = back_dists_mat[:,:,:-1] # N, L, L
            sub_pos_proto_dists = sub_pos_proto_dists * torch.eye(label_num) # N, L, L
            sub_pos_proto_dists = torch.sum(sub_pos_proto_dists, dim=-1) # N, L

    pred_pos_mask = (sub_pos_proto_dists > labed_pos_self_dists_mean).to(torch.float)
    dist_pred_pos_cnt = torch.sum(pred_pos_mask, dim=-1)  # N
    car_diff = torch.abs(dist_pred_pos_cnt - label_cardinality)
    aa = back_pred_pos_mask * (sub_pos_proto_dists + 1) / 2
    aa = 1 / torch.sum(aa + 1e-10, dim=-1)
    add_score_mat = torch.pow(aa, 0.5) * torch.pow(car_diff, 1 - 0.5)
    score_mat = add_score_mat.unsqueeze(-1)
    all_scores = torch.mean(score_mat, dim=-1)
    _, score_pair_indices = torch.topk(all_scores, len(all_scores))
    score_pair_indices = score_pair_indices[np.isin(score_pair_indices, label_wise_part_unlabeled)]
    if args.save_sub_tensor:
        ranked_indices = score_pair_indices + 0
        rank_id_path = os.path.join(this_record_path, 'rank_id')
        torch.save(ranked_indices, rank_id_path)
    query_pair_indices = score_pair_indices[:args.sample_pair_num]
    example_indices = query_pair_indices
    query_example_indices = [all_indices[int(x)] for x in example_indices]
    query_label_indices = None

    annotate_example_indices = torch.tensor([])
    annotate_label_indices = torch.tensor([])
    print('both len', len(query_example_indices), len(annotate_example_indices))
    return query_example_indices, query_label_indices, annotate_example_indices, annotate_label_indices



def query_samples_other(args, models, method, dataset, subset, device, label_num, label_cardinality=None, labeled_subset=None):
    for key in models.keys():
        models[key].eval()
    all_preds, all_indices = [], []
    all_ins_weights = []
    all_loss_weight = []
    all_mod_preds = []
    all_backbone_preds = []
    all_back_feats = []
    all_mmc_labels = []
    all_unlabed_v = []
    embedding = []
    if method in ['core_set', 'badge']:
        unlabeled_dataloader = DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          num_workers=NUM_WORKER,
                                          pin_memory=True,
                                          sampler=SubsetSequentialSampler(subset)
                                          )
    else:
        unlabeled_dataloader = DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          num_workers=NUM_WORKER,
                                          pin_memory=True,
                                          sampler=SubsetRandomSampler(subset)
                                          )
    if method in ['cvirs']:
        labeled_dataloader = DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          num_workers=NUM_WORKER,
                                          pin_memory=True,
                                          sampler=SubsetRandomSampler(labeled_subset)
                                          )
        all_labed_gt_labels = []
        all_labed_H2 = []
        for _, gt_label, idx, _, _ in tqdm(labeled_dataloader):
            all_labed_gt_labels.extend(gt_label)
            gt_pos_cnt = torch.sum(gt_label, dim=-1).to(torch.float) # N
            one_H2 = get_H2(gt_pos_cnt, label_num-gt_pos_cnt, label_num)
            all_labed_H2.extend(one_H2)
        all_labed_gt_labels = torch.stack(all_labed_gt_labels).to(torch.long)
        all_labed_H2 = torch.stack(all_labed_H2)
    for text_inputs, _, idx, loss_weight, ins_weight in tqdm(unlabeled_dataloader):
        input_ids, input_type_ids, input_mask = [x.to(device) for x in text_inputs]
        with torch.no_grad():
            preds, atten_out, encoder_cls = models['backbone']([input_ids, input_type_ids, input_mask, None])
            preds = preds.cpu().data
            all_backbone_preds.extend(preds)
            if method == 'lloss':
                mod_preds = models['loss_module'](encoder_cls)
                mod_preds = mod_preds.cpu().data
                all_mod_preds.extend(mod_preds)
            elif method == 'mmc':
                mmc_input = torch.sigmoid(preds)
                mmc_input, mmc_idx = torch.topk(mmc_input, label_num, dim=-1)
                mmc_input_sum = torch.sum(mmc_input, dim=-1)
                mmc_input = mmc_input / mmc_input_sum.unsqueeze(-1)
                mmc_input = mmc_input.to(device)
                mmc_output = models['mmc_module'](mmc_input).cpu().data
                mmc_pred_label_num = torch.topk(mmc_output, 1, dim=-1)[1] + 1
                mmc_pred_labels = torch.zeros_like(preds) - 1
                for j in range(preds.shape[0]):
                    pred_pos_idxs = torch.topk(preds[j], mmc_pred_label_num[j][0], dim=-1)[1]
                    mmc_pred_labels[j, pred_pos_idxs] = 1
                all_mmc_labels.extend(mmc_pred_labels)
            elif method == 'cvirs':
                pred_labels = (preds > 0).to(torch.long)
                pred_pos_mask = (preds > 0).to(torch.float)
                pred_neg_mask = 1 - pred_pos_mask
                cross_same_mask = (pred_labels.unsqueeze(1) == all_labed_gt_labels).to(torch.long) # N, La, L
                cross_same_pos_mask = cross_same_mask * pred_pos_mask.unsqueeze(1) # N, La, L
                cross_same_neg_mask = cross_same_mask * pred_neg_mask.unsqueeze(1) # N, La, L
                cross_diff_pos_mask = (1 - cross_same_mask) * pred_pos_mask.unsqueeze(1) # N, La, L
                cross_diff_neg_mask = (1 - cross_same_mask) * pred_neg_mask.unsqueeze(1) # N, La, L
                cross_same_pos_cnt = torch.sum(cross_same_pos_mask, dim=-1) # N, La
                cross_same_neg_cnt = torch.sum(cross_same_neg_mask, dim=-1) # N, La
                cross_diff_pos_cnt = torch.sum(cross_diff_pos_mask, dim=-1) # N, La
                cross_diff_neg_cnt = torch.sum(cross_diff_neg_mask, dim=-1) # N, La
                a = cross_same_pos_cnt
                d = cross_same_neg_cnt
                b = cross_diff_pos_cnt
                c = cross_diff_neg_cnt
                one_H4 = get_H4(a, b, c, d, label_num) # N, La

                cross_same_cnt = torch.sum(cross_same_mask, dim=-1) # N, La
                dH = label_num - cross_same_cnt # N, La
                dH_equal_1_mask = (dH == label_num).to(torch.float) # N, La
                pred_pos_cnt = torch.sum(pred_labels, dim=-1).to(torch.float)  # N
                one_H2 = get_H2(pred_pos_cnt, label_num - pred_pos_cnt, label_num) # N

                dE = 2 * one_H4 - all_labed_H2.unsqueeze(0) - one_H2.unsqueeze(-1) # N, La
                dE /= one_H4 + 1e-10
                fu = dH_equal_1_mask * 1 + (1 - dH_equal_1_mask) * dE # N, La
                one_v = torch.mean(fu, dim=-1)
                all_unlabed_v.extend(one_v)
        if method == 'core_set':
            encoder_cls = encoder_cls.cpu().data
            all_back_feats.extend(encoder_cls)
        elif method == 'badge':
            feats = encoder_cls.cpu().data
            pred_probs = torch.sigmoid(preds).data.cpu()
            pred_gts = torch.round(pred_probs)
            scale = pred_gts - pred_probs
            scale = torch.mean(scale, dim=-1)
            scale = scale.unsqueeze(-1)
            embs = scale * feats
            embedding.extend(embs)
        all_indices.extend(idx)
        all_loss_weight.extend(loss_weight)
        all_ins_weights.extend(ins_weight)
    all_indices = torch.stack(all_indices)
    # all_loss_weight = torch.stack(all_loss_weight)
    all_ins_weights = torch.stack(all_ins_weights)
    if method == 'lloss':
        all_mod_preds = torch.stack(all_mod_preds)
        lloss_preds = torch.mean(all_mod_preds, dim=-1)
        _, query_pair_indices = torch.topk(lloss_preds * (1 - all_ins_weights), len(all_mod_preds))
        query_pair_indices = query_pair_indices[:args.sample_pair_num]
    elif method == 'random':
        query_pair_indices = list(range(len(all_indices)))
        random.shuffle(query_pair_indices)
        query_pair_indices = query_pair_indices[:args.sample_pair_num]
    elif method == 'cvirs':
        all_backbone_preds = torch.stack(all_backbone_preds)
        all_unlabed_v = torch.stack(all_unlabed_v) # N
        all_backbone_preds = torch.sigmoid(all_backbone_preds)
        all_margin = torch.abs(all_backbone_preds - 0.5)
        margin_rank_id = all_margin.sort(0, True)[1]
        margin_rank_mat = torch.zeros_like(all_margin, dtype=torch.long)
        label_ids = torch.tensor(list(range(label_num))).unsqueeze(0)
        margin_rank_mat[margin_rank_id, label_ids] = torch.tensor(list(range(len(all_margin)))).unsqueeze(-1)
        margin_rank_mat = len(all_margin) - margin_rank_mat
        margin_rank_mat = margin_rank_mat.to(torch.float) / (label_num * (len(all_margin) - 1))
        margin_rank_mat = torch.sum(margin_rank_mat, dim=-1) # N
        score_mat = margin_rank_mat * all_unlabed_v
        _, query_pair_indices = torch.topk(score_mat * (1 - all_ins_weights), len(score_mat))
        query_pair_indices = query_pair_indices[:args.sample_pair_num]
    elif method == 'mmc':
        all_backbone_preds = torch.stack(all_backbone_preds)
        all_backbone_preds = torch.tanh(all_backbone_preds)
        all_mmc_labels = torch.stack(all_mmc_labels)
        score_mat = -1 * all_mmc_labels * all_backbone_preds + 1
        score_mat = torch.sum(score_mat, dim=-1)
        _, query_pair_indices = torch.topk(score_mat * (1 - all_ins_weights), len(all_backbone_preds))
        query_pair_indices = query_pair_indices[:args.sample_pair_num]
    elif method == 'adaptive':
        all_backbone_preds = torch.stack(all_backbone_preds)
        all_backbone_preds = torch.sigmoid(all_backbone_preds)
        pred_pos_mask = (all_backbone_preds >= 0.5).to(torch.int)
        pred_neg_mask = (all_backbone_preds < 0.5).to(torch.int)
        pred_neg_mat = all_backbone_preds * pred_neg_mask
        pred_neg_mat_max = torch.max(pred_neg_mat, dim=-1)[0]
        pred_pos_mat = all_backbone_preds * pred_pos_mask + pred_neg_mask * 2
        pred_pos_mat_min = torch.min(pred_pos_mat, dim=-1)[0]
        pred_margin = pred_pos_mat_min - pred_neg_mat_max # N
        pred_margin = 1 / pred_margin
        pred_pos_cnt = torch.sum(pred_pos_mask, dim=-1) # N
        car_diff = torch.abs(pred_pos_cnt - label_cardinality) # N
        if args.adaptive_b_mode == 0:
            B_list = torch.range(0,1,0.1)
        elif args.adaptive_b_mode == 1:
            B_list = torch.tensor([0.5])
        top_idx_mat = []
        cnt_len = args.sample_pair_num // len(B_list)
        query_pair_indices = []
        for b in B_list:
            score_mat = torch.pow(pred_margin, b) * torch.pow(car_diff, 1-b) * (1 - all_ins_weights)
            top_idx = torch.topk(score_mat, len(score_mat))[1]
            top_idx_mat.append(top_idx)
            query_pair_indices.extend(top_idx[:cnt_len].numpy().tolist())
        query_pair_indices = list(set(query_pair_indices))
        top_idx_mat = torch.stack(top_idx_mat)
        j = cnt_len
        while len(query_pair_indices) < args.sample_pair_num:

            for i in range(len(B_list)):
                if top_idx_mat[i][j] not in query_pair_indices:
                    query_pair_indices.append(top_idx_mat[i][j])
            j += 1
        query_pair_indices = query_pair_indices[:args.sample_pair_num]
        query_pair_indices = torch.tensor(query_pair_indices)
    elif method == 'core_set':
        all_back_feats = torch.stack(all_back_feats).numpy()
        sampling = kCenterGreedy(all_back_feats)
        flat_loss_weight = all_ins_weights.view(-1)
        av_idx = torch.nonzero(flat_loss_weight == 1)
        new_av_idx = [x[0] for x in av_idx]
        query_pair_indices = sampling.select_batch_(new_av_idx, args.sample_pair_num)
    elif method == 'badge':
        embedding = torch.stack(embedding)
        flat_loss_weight = all_ins_weights.view(-1)
        already_selected_idx = torch.nonzero(flat_loss_weight == 1)
        already_selected_idx = [x[0] for x in already_selected_idx]
        query_pair_indices = init_centers(embedding.numpy(), args.sample_pair_num, already_selected_idx)
    query_example_indices = [all_indices[x] for x in query_pair_indices]
    query_label_indices = None
    annotate_example_indices = []
    annotate_label_indices = []

    return query_example_indices, query_label_indices, annotate_example_indices, annotate_label_indices


def init_centers(X, K, already_selected):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    # while len(mu) < K:
    for _ in tqdm(range(1,K)):
        if len(mu) >= K:
            break
        if len(already_selected) + len(indsAll) == len(X):
            break
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll or ind in already_selected:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def pred_part_stat(pred, thresold=0.0):
    # pred_round = torch.round(pred)
    part_1 = torch.where(pred >= thresold)[0]
    part_0 = torch.where(pred < thresold)[0]
    return part_0, part_1






