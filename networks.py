from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel, BertConfig

class BackBone_No_GCN_No_Atten(nn.Module):
    def __init__(self, args, label_num):
        super(BackBone_No_GCN_No_Atten, self).__init__()
        self.args = args
        self.label_num = label_num
        model_config = BertConfig.from_pretrained(args.bert_path)
        self.hidden_size = model_config.hidden_size * self.args.feature_layer

        self.encoder_no_gcn_no_atten = Encoder_No_GCN_No_Atten(args, label_num)
        self.clf = nn.Linear(model_config.hidden_size, label_num)
        nn.init.xavier_uniform_(self.clf.weight)

    def forward(self, inputs):
        encoder_out = self.encoder_no_gcn_no_atten(inputs)
        out = self.clf(encoder_out)
        return out, encoder_out, encoder_out

class Encoder_No_GCN_No_Atten(nn.Module):
    def __init__(self, args, label_num):
        super(Encoder_No_GCN_No_Atten, self).__init__()
        self.args = args
        self.label_num = label_num
        model_config = BertConfig.from_pretrained(args.bert_path)
        self.hidden_size = model_config.hidden_size * self.args.feature_layer
        self.encoder = BertModel.from_pretrained(args.bert_path, config=model_config)
        self.encoder_init()

    def encoder_init(self):
        all_layers = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6',
                      'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler']
        if self.args.freeze_bert:
            unfreeze_layers = all_layers[self.args.freeze_layer_num + 1:]
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break

    def forward(self, inputs):
        input_ids, input_type_ids, input_mask, _ = inputs
        out_list = self.encoder(input_ids=input_ids,
                                token_type_ids=input_type_ids,
                                attention_mask=input_mask)
        last_hidden_state, cls = out_list[0], out_list[1]
        return cls

class MLLinear(nn.Module):
    def __init__(self, state_list, output_size):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)


class LLoss_Module(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LLoss_Module, self).__init__()
        self.clf = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.clf.weight)

    def forward(self, inputs):
        out = self.clf(inputs)
        return out

class MMC_Module(nn.Module):
    def __init__(self, label_num):
        super(MMC_Module, self).__init__()
        self.linear = nn.Linear(label_num, label_num)
        nn.init.xavier_uniform_(self.linear.weight)
        self.sm = nn.Softmax()

    def forward(self, inputs):
        out = self.linear(inputs)
        out = self.sm(out)
        return out


class MLP_VAE(nn.Module):
    def __init__(self, args, hidden_size, device):
        super(MLP_VAE, self).__init__()
        self.args = args
        self.device = device
        self.label_num = args.label_num
        self.fc0 = nn.Linear(args.hidden_size, args.label_num * 512)
        self.fc1 = nn.Linear(512, args.proto_size)

        if args.cl_neg_mode == 0:
            cl_label_num = args.label_num
        elif args.cl_neg_mode == 1:
            cl_label_num = args.label_num + 1
            self.ins_neg_cnt = torch.zeros(1).to(device)
        elif args.cl_neg_mode == 2:
            cl_label_num = args.label_num * 2
            self.ins_neg_cnt = torch.zeros(args.label_num).to(device)
        self.cl_label_num = cl_label_num
        self.register_buffer("prototypes", torch.zeros(self.cl_label_num, args.proto_size))
        self.ins_pos_cnt = torch.zeros(args.label_num).to(device)

        self.fc3 = nn.Linear(args.proto_size, 512)
        self.agg = nn.Linear(args.label_num * 512, args.hidden_size)
        self.clf = nn.Linear(args.hidden_size, args.label_num)

    def get_protos(self):
        return self.prototypes

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reset_proto(self):
        self.ins_pos_cnt *= 0
        self.ins_neg_cnt *= 0
        self.prototypes *= 0

    def forward(self, x, labels=None, weights=None):
        sub_rep_ori = self.fc0(x)
        sr_shape = sub_rep_ori.shape
        sub_rep_ori = sub_rep_ori.view(sr_shape[0], self.args.label_num, -1)
        sub_rep = self.fc1(sub_rep_ori)
        dec_sub_rep = self.fc3(sub_rep)
        dist_1 = None
        sub_rep_norm = F.normalize(sub_rep.data, dim=-1)
        if labels is not None and weights is not None:
            pos_mask = labels * weights
            neg_mask = (1 - labels) * weights
            pos_mask = pos_mask.unsqueeze(-1)
            neg_mask = neg_mask.unsqueeze(-1)
            feat = torch.sum(sub_rep_norm * pos_mask, dim=0)
            self.prototypes[:self.label_num] = self.prototypes[:self.label_num] * self.ins_pos_cnt.unsqueeze(-1) + feat
            if self.args.cl_neg_mode == 2:
                feat_neg = torch.sum(sub_rep_norm * neg_mask, dim=0)
                self.prototypes[self.label_num:] = self.prototypes[self.label_num:] * self.ins_neg_cnt.unsqueeze(
                    -1) + feat_neg
                self.ins_neg_cnt += torch.sum(neg_mask.squeeze(-1), dim=0)
            elif self.args.cl_neg_mode == 1:
                feat_neg = torch.sum(sub_rep_norm * neg_mask, dim=0)
                feat_neg = torch.sum(feat_neg, dim=0)
                self.prototypes[-1] = self.prototypes[-1] * self.ins_neg_cnt + feat_neg
                self.ins_neg_cnt += torch.sum(neg_mask)
            self.ins_pos_cnt += torch.sum(pos_mask.squeeze(-1), dim=0)
            self.prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        if self.args.cl_neg_mode == 2:
            dist_1 = torch.einsum('bld,ld->bl', [sub_rep_norm, self.prototypes[:self.label_num]])
            dist_neg_1 = torch.einsum('bld,ld->bl', [sub_rep_norm, self.prototypes[self.label_num:]])
            dist_1 = torch.cat([dist_1, dist_neg_1], dim=1)
        elif self.args.cl_neg_mode == 0 or self.args.cl_neg_mode == 1:
            dist_1 = sub_rep_norm @ self.prototypes.T
        concat_sub_rep = dec_sub_rep.contiguous().view(dec_sub_rep.shape[0], -1)
        recon_x = self.agg(concat_sub_rep)
        recon_x_clf = self.clf(recon_x)
        return recon_x, sub_rep, recon_x_clf, dist_1







