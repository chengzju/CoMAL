import torch
import torch.nn as nn
import torch.nn.functional as F




def clf_loss(y_pred, y_true, mask=None, eps=1e-10,
             label_weight=None, pos_weight=None):
    if mask is not None:
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_fct(y_pred, y_true)
        loss = torch.sum(loss * mask) / (torch.sum(mask) + eps)
    else:
        if label_weight is not None:
            label_weight = label_weight.to(y_true.device)
        if pos_weight is not None:
            pos_weight = pos_weight.to(y_true.device)
        loss_fct = nn.BCEWithLogitsLoss(label_weight, pos_weight=pos_weight)
        loss = loss_fct(y_pred, y_true)
    return loss



def LossPredLoss(input, target, margin=1.0, loss_weight=None, reduction='mean', labels=None):
    input = input.view(-1)
    target = target.view(-1)

    if loss_weight is not None:
        loss_weight = loss_weight.view(-1)
        input = input[loss_weight > 0.5]
        target = target[loss_weight > 0.5]
    if labels is not None:
        labels = labels.view(-1)
        labels = labels[loss_weight > 0.5]

    if len(input) % 2 != 0:
        input = input[:-1]
        target = target[:-1]
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss



def vae_loss(x, recon_x, mu, logvar, beta=1):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon_x, x)
    if mu is not None and logvar is not None:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        MSE += KLD
    return MSE


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, neg_mask=None, batch_size=-1, device=None, other_features=None):

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach()
            # compute logits
            if other_features is None:
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
            else:
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], other_features.T),
                    self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            if neg_mask is None:
                logits_mask = torch.ones_like(mask)
            else:
                logits_mask = torch.scatter(
                    neg_mask,
                    1,
                    torch.arange(batch_size).view(-1, 1).to(device),
                    0
                )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss