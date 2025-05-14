import torch
from torch import nn



# Simplified weighted tuple loss with dynamic negative sampling
class WBLossWithDynamicWeights(nn.Module):
    def __init__(self, alpha=10, **kwargs):
        super(WBLossWithDynamicWeights, self).__init__()
        self.alpha = alpha

    def forward(self, inputs_q, inputs_k):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q)
        
        return (loss_1 + loss_2) * 0.5, (mean_pos_sim_1 + mean_pos_sim_2) * 0.5, (mean_neg_sim_1 + mean_neg_sim_2) * 0.5

    def single_forward(self, inputs_q, inputs_k):
        n = inputs_q.size(0)

        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)

        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())  # [n, n]

        eyes_ = torch.eye(n).cuda()
        pos_mask = eyes_.eq(1)  # 正样本对角线
        neg_mask = ~pos_mask   # 负样本非对角线

        pos_sim = torch.masked_select(sim_mat, pos_mask)  # [n]
        neg_sim = torch.masked_select(sim_mat, neg_mask).reshape(n, n - 1)  # [n, n-1]

        neg_weights = torch.softmax(neg_sim, dim=1)  # 动态权重，形状 [n, n-1]

        neg_sim_dynamic = neg_weights * neg_sim * neg_sim.size(1)  # 动态权重放大回去

        
        pos_sim_ = pos_sim.unsqueeze(dim=1).expand(n, n - 1)
        neg_sim_ = neg_sim_dynamic.reshape(n, n - 1)


        loss_batch = torch.log(1 + torch.sum(torch.exp((neg_sim_ - pos_sim_) * self.alpha), dim=1))

        # 检查数值稳定性
        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch.mean()

        # 返回正负样本的均值相似度和损失
        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()

        return loss, mean_pos_sim, mean_neg_sim

