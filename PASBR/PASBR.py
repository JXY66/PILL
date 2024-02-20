import math

import torch as th
from torch import nn
import dgl
import dgl.ops as F
import dgl.function as fn
import numpy as np
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EOPA(nn.Module):
    def __init__(
            self, input_dim, output_dim, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        """
        计算来自所有邻居节点的聚合信息
        """
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        # since the edges are sorted by occurrence time when the EOP multigraph is built
        # the messages are in the order required by EOPA
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = self.feat_drop(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']

                # 当前节点本身的特征+邻居节点聚合后的特征 -> 得到当前节点的新特征
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        # 计算每个邻居节点对当前节点的权重
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        # 计算边的特征: q -> 源节点特征 k -> 目标节点特征
        # 实际上就是将邻居节点的权重当作当前节点对应入边上的边特征
        e = F.u_add_v(sg, q, k)


        e = self.fc_e(th.sigmoid(e))

        # 权重归一化.
        a = F.edge_softmax(sg, e)

        # 加权聚合邻居节点的表示来更新当前节点
        rst = F.u_mul_e_sum(sg, v, a)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        # fc_i: 用于最后一个节点嵌入
        self.fc_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation

    def forward(self, g, feat, intend, last_nodes, position_weight):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        # feat: [n, 160]
        feat = self.feat_drop(feat)

        # feat_u: [n, 32]
        feat_u = self.fc_u(feat)
        # W*x_last + r

        # feat_v: [n, 32]
        # intend: [512, 160]
        feat_v = self.fc_v(intend)
        feat_v = dgl.broadcast_nodes(g, feat_v)

        # 保留最后一个节点嵌入
        # feat[last_nodes]: [512, 160]
        feat_last = self.fc_i(feat[last_nodes])
        # feat_last: [n, 32]
        feat_last = dgl.broadcast_nodes(g, feat_last)

        # e表示每个商品与最后一个商品计算出来的重要分数
        # e: [n, 1]
        e = self.fc_e(th.sigmoid(feat_u + feat_v + feat_last))

        # alpha: [n, 1]
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        # 对每个商品嵌入进行加权求和
        # feat_norm: [n, 160]
        feat_norm = feat * alpha

        # rst: [512, 32]
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)

        # TODO:新增: 考虑物品之间的相对位置权重,并对物品嵌入进行加权求和
        position_weight = position_weight.unsqueeze(1)
        feat_postion = feat * position_weight
        position_rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_postion, 'sum').type(th.float32)
        if self.fc_out is not None:
            position_rst = self.fc_out(position_rst)
        if self.activation is not None:
            position_rst = self.activation(position_rst)
        return rst, position_rst


class AttnReadout_without_intent(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        # fc_i: 用于最后一个节点嵌入
        self.fc_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation

    # 原: 利用最后一个节点计算每个节点重要性并进行聚合得到s_g
    # TODO:消融1:无意图
    def forward(self, g, feat, last_nodes, position_weight):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        # W*x_i
        feat_u = self.fc_u(feat)
        # W*x_last + r
        # 保留最后一个节点嵌入
        feat_last = self.fc_i(feat[last_nodes])
        feat_last = dgl.broadcast_nodes(g, feat_last)
        e = self.fc_e(th.sigmoid(feat_u + feat_last))

        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        # 对每个商品嵌入进行加权求和
        feat_norm = feat * alpha

        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)

        # TODO:新增: 考虑物品之间的相对位置权重,并对物品嵌入进行加权求和
        position_weight = position_weight.unsqueeze(1)
        feat_postion = feat * position_weight
        position_rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_postion, 'sum').type(th.float32)
        if self.fc_out is not None:
            position_rst = self.fc_out(position_rst)
        if self.activation is not None:
            position_rst = self.activation(position_rst)

        return rst, position_rst


class Intend(nn.Module):
    def __init__(
            self, input_dim, output_dim, seq_len, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(seq_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, output_dim, batch_first=False)
        self.activation = activation
        self.input_dim = input_dim

    def forward(self, feat, hidden):

        _, hidden = self.gru(feat, hidden)

        # if self.activation is not None:
        #     hidden = self.activation(hidden)
        return hidden.squeeze(0)


class PriceAware(nn.Module):
    def __init__(
            self, input_dim, output_dim, batch_norm=True, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.price_factor = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        self.input_dim = input_dim

    def forward(self, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        price_factor = self.price_factor(feat)
        if self.activation is not None:
            price_factor = self.activation(price_factor)
        return price_factor


class PILL(nn.Module):
    def __init__(
            self, args, num_items, num_category, num_price, embedding_dim, num_layers, batch_norm=True, feat_drop=0.0
    ):
        super().__init__()
        self.args = args
        if self.args.without_intent == 1 and self.args.without_price == 0:
            self.batch_norm_without_intent = True
        else:
            self.batch_norm_without_intent = False
        if self.args.without_intent == 1 and self.args.without_price == 1:
            self.batch_norm_origin = True
        else:
            self.batch_norm_origin = False
        self.embedding_dim = embedding_dim
        self.embedding_dim_2 = embedding_dim + embedding_dim

        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1.5)
        # TODO:初始化类别嵌入矩阵
        self.category_embedding = nn.Embedding(num_category, embedding_dim, max_norm=1.5)
        # TODO:初始化价格嵌入矩阵
        self.price_embedding = nn.Embedding(num_price, embedding_dim, max_norm=1.5)
        self.pos_embedding = nn.Embedding(21, embedding_dim, max_norm=1.5)
        # TODO: 初始化位置索引嵌入矩阵(序列长度最长为20,因此设为21)
        self.index_embedding = nn.Embedding(21, embedding_dim, max_norm=1.5)
        # TODO:转换价格和类别嵌入的级联矩阵
        self.price_category = nn.Linear(self.embedding_dim_2, self.embedding_dim, bias=False)

        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim

        # TODO:新增自注意力机制层
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.embedding_dim, nhead=args.nhead,
                                                               dim_feedforward=self.embedding_dim * args.feedforward)
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, args.layer)

        # 设置意图GRU的输出维度: 160
        intend_output_dim = input_dim * (num_layers + 1)
        self.hidden_size = intend_output_dim
        # price_user_output_dim = input_dim * num_layers
        self.intend = Intend(input_dim,
                             intend_output_dim,
                             seq_len=20,
                             batch_norm=batch_norm,
                             feat_drop=feat_drop,
                             activation=nn.ReLU(),
                             )
        self.price_aware = PriceAware(
                            input_dim,
                            output_dim=intend_output_dim,
                            batch_norm=batch_norm,
                            activation=nn.Sigmoid()
                            )
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    batch_norm=batch_norm,
                    feat_drop=feat_drop,
                    activation=nn.PReLU(embedding_dim),
                )
            # 因为下一层的输入等于先前各层的输出级联
            # 因此下一层的输入维度会进行累加
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        self.readout_without_intent = AttnReadout_without_intent(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.PReLU(embedding_dim),
        )
        input_dim += embedding_dim

        # 放开这一行代码,用于运行动态兴趣模块的消融实验
        # fc_sr_input_dim = input_dim + intend_output_dim
        # 注释掉这一行代码，用于运行动态兴趣模块的消融实验
        fc_sr_input_dim = input_dim + intend_output_dim + embedding_dim
        # TODO:消融1:无意图
        fc_sr_input_dim_without_intent = input_dim + embedding_dim

        self.batch_norm_without_intent = nn.BatchNorm1d(fc_sr_input_dim_without_intent) if self.batch_norm_without_intent else None
        self.batch_norm = nn.BatchNorm1d(fc_sr_input_dim) if batch_norm else None
        self.batch_norm_final = nn.BatchNorm1d(intend_output_dim) if batch_norm else None
        self.batch_norm_origin = nn.BatchNorm1d(input_dim) if self.batch_norm_origin else None
        self.feat_drop = nn.Dropout(feat_drop)
        # 288 -> 32
        self.fc_sr = nn.Linear(fc_sr_input_dim, embedding_dim, bias=False)
        self.fc_sr_without_intent = nn.Linear(fc_sr_input_dim_without_intent, embedding_dim, bias=False)
        self.fc_sr_origin = nn.Linear(input_dim, embedding_dim, bias=False)
        # 160 -> 32
        self.fc_sr_final = nn.Linear(intend_output_dim, embedding_dim, bias=False)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.LayerNorm = nn.LayerNorm(self.embedding_dim, eps=1e-8)
        self.dropout = nn.Dropout(0.25)

    def init_hidden(self, batch_size):
        return th.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, mg, sg=None):
        device = th.device('cuda')
        # 获取包含每个序列图的list
        # batch_num_nodes():每张图的无重复节点数
        # each_graph_nodes: tensor 512
        each_graph_nodes = mg.batch_num_nodes()

        # 所有样本序列的原序类别ID序列
        all_graph_origin_cid = []

        # 获取每个序列的顺序类别ID序列
        # 记录当前循环的末尾索引
        end_index = 0
        # 记录所有末尾索引,用于下一次循环开始时的起始索引
        all_end_index = [0] * len(each_graph_nodes)
        for i, j in enumerate(each_graph_nodes):
            end_index += j
            if i != len(each_graph_nodes) - 1:
                all_end_index[i+1] = int(end_index)
            from_index = all_end_index[i]
            each_graph_origin_cid = []
            for index in list(range(from_index, end_index)):
                each_graph_origin_cid.append(mg.ndata['origin_cid'][index].long())
            all_graph_origin_cid.append(each_graph_origin_cid)
        sorted_all_graph_origin_cid = sorted(all_graph_origin_cid, key=lambda x: len(x), reverse=True)
        # 获取商品ID、类别ID、价格ID
        iid = mg.ndata['iid']
        iid = iid.long()

        cid = mg.ndata['cid']
        cid = cid.long()

        # TODO:获取价格ID序列
        pid = mg.ndata['pid']
        pid = pid.long()

        index2 = mg.ndata['index_info']
        index2 = index2.long()

        position_weight = mg.ndata['position_weight']

        # 获取商品的初始嵌入[node_num, dim]   自注意力需要输入: [batch_size, seq_len, dim]
        feat = self.embedding(iid)

        # 获取商品的价格嵌入
        price_embedding = self.price_embedding(pid)
        # 获取商品的类别初始嵌入
        category_feat = self.category_embedding(cid)
        # 获取位置索引嵌入
        index_feat2 = self.index_embedding(index2)

        # 位置索引嵌入 + ID初始嵌入
        # feat = feat + index_feat2
        # feat = self.LayerNorm(feat)
        # feat = self.dropout(feat)

        if self.args.without_intent == 1 and self.args.without_price == 0:
            # TODO:消融实验1: 不考虑意图特征
            # TODO:与类别嵌入计算权重
            # 融合价格初始嵌入矩阵和类别初始嵌入矩阵的各种方式
            # 两者进行相加
            price_category = price_embedding.add_(category_feat)

            price_factor = self.price_aware(price_category)

            for i, layer in enumerate(self.layers):
                if i % 2 == 0:
                    out = layer(mg, feat)
                else:
                    out = layer(sg, feat)

                # 将每一层的输出进行级联作为下一层的输入
                feat = th.cat([out, feat], dim=1)

            # 更新完毕,feat表示已经更新后的节点嵌入矩阵
            # 获取最后一个节点的索引
            # 不仅仅是一个,而是一个batch_size样本的最后一个节点
            # batch_size=512,因此总共是512个序列中最后一个节点索引组成的一维张量
            last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)

            # 获取全局偏好表示.此时的feat表示已经更新后的所有节点的表示矩阵
            # feat的特征维度已经变成160维,因为是原始输入维度+4个层的输出(将每一层的输出进行水平级联
            # sr_g: [512, 32] 512表示batch_size
            # TODO: 用价格表示更新节点表示
            feat = price_factor.mul(feat)
            # TODO:消融2:无意图
            sr_g, pos_g = self.readout_without_intent(mg, feat, last_nodes, position_weight)

            # 获取最后一个节点经过多层学习后的表示,size=[batch_size, 160]
            sr_l = feat[last_nodes]

            # TODO:消融2:无意图
            sr = th.cat([sr_l, sr_g, pos_g], dim=1)

            if self.batch_norm_without_intent is not None:
                sr = self.batch_norm_without_intent(sr)
            sr = self.fc_sr_without_intent(self.feat_drop(sr))
            logits = sr @ self.embedding(self.indices).t()
            return logits
        elif self.args.without_intent == 0 and self.args.without_price == 1:
            # TODO: 消融实验2: 无价格因子
            # 一个batch的类别序列矩阵
            input_cate_matrix = th.zeros((len(all_graph_origin_cid), 20, self.embedding_dim))

            for i, each_seq in enumerate(all_graph_origin_cid):
                fill_intend_matrix = th.zeros((20, self.embedding_dim))
                each_seq = th.from_numpy(np.array(each_seq).astype(dtype=np.int32)).long().to(device)
                # seqs_lens.append(len(each_seq))
                each_cate_feat = self.category_embedding(each_seq)
                # 类别序列进行零填充
                if len(each_cate_feat) < 20:
                    for j in list(range(1, len(each_cate_feat) + 1))[::-1]:
                        fill_intend_matrix[-j] = each_cate_feat[-j]
                # 将每个填充后的类别序列堆叠成一个三维矩阵[batch_size, seq_len, dim]
                input_cate_matrix[i] = fill_intend_matrix
            # TODO:消融2:无价格
            # input_cate_matrix: [512, 20, 32]
            input_cate_matrix = input_cate_matrix.to(device)
            batch_size = input_cate_matrix.shape[0]

            hidden = self.init_hidden(batch_size)
            intend_matrix = self.intend(input_cate_matrix, hidden)
            # [512, 160]
            intend_matrix = intend_matrix.to(device)

            # 多层的交替更新
            for i, layer in enumerate(self.layers):
                if i % 2 == 0:
                    out = layer(mg, feat)
                else:
                    out = layer(sg, feat)

                # 将每一层的输出进行级联作为下一层的输入
                feat = th.cat([out, feat], dim=1)

            # 更新完毕,feat表示已经更新后的节点嵌入矩阵
            # 获取最后一个节点的索引
            # 不仅仅是一个,而是一个batch_size样本的最后一个节点
            # batch_size=512,因此总共是512个序列中最后一个节点索引组成的一维张量
            last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)

            # 获取全局偏好表示.此时的feat表示已经更新后的所有节点的表示矩阵
            # feat的特征维度已经变成160维,因为是原始输入维度+4个层的输出(将每一层的输出进行水平级联)
            # sr_g: [512, 32] 512表示batch_size

            sr_g, pos_g = self.readout(mg, feat, intend_matrix, last_nodes, position_weight)

            # 获取最后一个节点经过多层学习后的表示,size=[batch_size, 160]
            sr_l = feat[last_nodes]
            # 获取会话整体表示
            sr = th.cat([intend_matrix, sr_l, sr_g, pos_g], dim=1)

            if self.batch_norm is not None:
                sr = self.batch_norm(sr)
            sr = self.fc_sr(self.feat_drop(sr))

            # 利用会话表示与商品的初始嵌入计算推荐分数 [512, 42596]
            # self.indices是大小为42596的一维张量,[0,1,2,...,42594,42595],表示所有商品的索引
            logits = sr @ self.embedding(self.indices).t()
            return logits
        elif self.args.without_intent == 1 and self.args.without_price == 1:
            for i, layer in enumerate(self.layers):
                if i % 2 == 0:
                    out = layer(mg, feat)
                else:
                    out = layer(sg, feat)

                # 将每一层的输出进行级联作为下一层的输入
                feat = th.cat([out, feat], dim=1)

            # 更新完毕,feat表示已经更新后的节点嵌入矩阵
            # 获取最后一个节点的索引
            # 不仅仅是一个,而是一个batch_size样本的最后一个节点
            # batch_size=512,因此总共是512个序列中最后一个节点索引组成的一维张量
            last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)

            # 获取全局偏好表示.此时的feat表示已经更新后的所有节点的表示矩阵
            # feat的特征维度已经变成160维,因为是原始输入维度+4个层的输出(将每一层的输出进行水平级联)
            # sr_g: [512, 32] 512表示batch_size
            # TODO: 用价格表示更新节点表示
            sr_g = self.readout(mg, feat, last_nodes)

            # 获取最后一个节点经过多层学习后的表示,size=[batch_size, 160]
            sr_l = feat[last_nodes]
            # 获取会话整体表示,sr: [512, 192]
            sr = th.cat([sr_l, sr_g], dim=1)

            if self.batch_norm_origin is not None:
                sr = self.batch_norm_origin(sr)
            sr = self.fc_sr_origin(self.feat_drop(sr))

            # 利用会话表示与商品的初始嵌入计算推荐分数 [512, 42596]
            # self.indices是大小为42596的一维张量,[0,1,2,...,42594,42595],表示所有商品的索引
            logits = sr @ self.embedding(self.indices).t()
            return logits
        else:
            # TODO:包括意图和价格特征
            # TODO:与类别嵌入计算权重
            # 融合价格初始嵌入矩阵和类别初始嵌入矩阵的各种方式
            # 1.直接进行按位相乘
            # price_category = price_embedding.mul(category_feat)
            # 3. 两者进行相加
            price_category = price_embedding.add_(category_feat)
            # 2. 两者进行级联再降维到32
            #price_category = th.cat((price_embedding, category_feat), 1)
            #price_category = self.price_category(price_category)
            # 4. 融合门控的方式
            # q_p = self.w_price(price_embedding)
            # q_c = self.w_category(category_feat)
            # f = th.sigmoid(q_p + q_c)
            # one_m = th.ones((item_nums, item_dim)).to(device)
            # price_category = f.mul(price_embedding).add_((one_m - f).mul(category_feat))

            price_factor = self.price_aware(price_category)

            # 每个序列获取一个意图表示,将其构成一个意图矩阵[batch_size, 160]
            # 用以替代s_l,即最后一个节点的表示矩阵
            # TODO: 根据类别嵌入序列获取意图表示
            # 一个batch的类别序列矩阵
            input_cate_matrix = th.zeros((len(all_graph_origin_cid), 20, self.embedding_dim))
            # 获取每条序列长度的列表
            seqs_lens = []
            for i, each_seq in enumerate(sorted_all_graph_origin_cid):
                each_seq = th.from_numpy(np.array(each_seq).astype(dtype=np.int32)).long().to(device)
                seqs_lens.append(len(each_seq))
                each_cate_feat = self.category_embedding(each_seq)
                fill_lens = 20 - len(each_seq)
                pad = nn.ZeroPad2d(padding=(0, 0, fill_lens, 0))
                pad_res = pad(each_cate_feat)
                # 将每个填充后的类别序列堆叠成一个三维矩阵[batch_size, seq_len, dim]
                input_cate_matrix[i] = pad_res
            # input_cate_matrix: [512, 20, 32]
            input_cate_matrix = input_cate_matrix.to(device)
            # TODO: 2205191635-添加位置嵌入并采用自注意力机制 -- 可能有提升
            # position_ids = th.arange(20, dtype=th.long, device=input_cate_matrix.device)
            # position_ids = position_ids.unsqueeze(0).expand((input_cate_matrix.shape[0], 20))
            # position_emb = self.pos_embedding(position_ids)
            # input_cate_matrix = input_cate_matrix + position_emb
            # input_cate_matrix = self.LayerNorm(input_cate_matrix)
            # input_cate_matrix = self.dropout(input_cate_matrix)
            # 33.99 68.95 无SAN
            # input_cate_matrix = input_cate_matrix.transpose(0, 1).contiguous()
            # input_cate_matrix = self.transformerEncoder(input_cate_matrix)
            # input_cate_matrix = input_cate_matrix.transpose(0, 1).contiguous()

            batch_size = input_cate_matrix.shape[0]
            input_cate_matrix = input_cate_matrix.permute(1, 0, 2)
            # 对填充后的类别嵌入矩阵进行压缩,压缩掉无效的填充值
            input_cate_matrix = pack_padded_sequence(input_cate_matrix, seqs_lens)
            hidden = self.init_hidden(batch_size)
            intend_matrix = self.intend(input_cate_matrix, hidden)
            # [512, 160]
            intend_matrix = intend_matrix.to(device)

            # 多层的交替更新
            for i, layer in enumerate(self.layers):
                if i % 2 == 0:
                    out = layer(mg, feat)
                else:
                    out = layer(sg, feat)

                # 将每一层的输出进行级联作为下一层的输入
                feat = th.cat([out, feat], dim=1)

            # 更新完毕,feat表示已经更新后的节点嵌入矩阵
            # 获取最后一个节点的索引
            # 不仅仅是一个,而是一个batch_size样本的最后一个节点
            # batch_size=512,因此总共是512个序列中最后一个节点索引组成的一维张量
            last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)

            # TODO: 额外采用意图表示来获取长期偏好特征sr_g
            # sr_g: [512, 32] 512表示batch_size
            # TODO: 用价格表示更新节点表示
            feat = price_factor.mul(feat)
            # 获取全局偏好表示.此时的feat表示已经更新后的所有节点的表示矩阵
            # feat的特征维度已经变成160维,因为是原始输入维度+4个层的输出(将每一层的输出进行水平级联)
            sr_g, pos_g = self.readout(mg, feat, intend_matrix, last_nodes, position_weight)

            # 获取最后一个节点经过多层学习后的表示,size=[batch_size, 160]
            sr_l = feat[last_nodes]
            # 获取会话整体表示,sr: [512, 192]
            sr = th.cat([intend_matrix, sr_l, sr_g, pos_g], dim=1)
            # 注释掉第678行代码，放开680行代码，验证动态兴趣模块（第三个消融实验）的有效性
            # sr = th.cat([intend_matrix, sr_l, sr_g], dim=1)
            if self.batch_norm is not None:
                sr = self.batch_norm(sr)
            sr = self.fc_sr(self.feat_drop(sr))

            # 利用会话表示与商品的初始嵌入计算推荐分数 [512, 42596]
            # self.indices是大小为42596的一维张量,[0,1,2,...,42594,42595],表示所有商品的索引
            logits = sr @ self.embedding(self.indices).t()
            return logits
