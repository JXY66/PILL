import torch as th
from torch import nn
import dgl
import dgl.ops as F
import dgl.function as fn
import numpy as np


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
        # 公式(10):计算每个邻居节点对当前节点的权重
        q = self.fc_q(feat)
        k = self.fc_k(feat)
        v = self.fc_v(feat)
        # 计算边的特征: q -> 源节点特征 k -> 目标节点特征
        # 实际上就是将邻居节点的权重当作当前节点对应入边上的边特征
        # 因此可以考虑将时间因素引入在该特征上
        e = F.u_add_v(sg, q, k)

        # TODO: 1.可以在计算邻居权重时考虑时间差的影响
        e = self.fc_e(th.sigmoid(e))

        # 公式(9): 权重归一化.
        a = F.edge_softmax(sg, e)

        # 加权聚合邻居节点的表示来更新当前节点,公式(8)
        # TODO: 2.或是将时间差因素体现在此处: a表示边的特征 v表示源节点特征
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
        self.fc_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation

    # 原: 利用最后一个节点计算每个节点重要性并进行聚合得到s_g
    # def forward(self, g, feat, last_nodes):
    def forward(self, g, feat, intend, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        # W*x_i
        feat_u = self.fc_u(feat)
        # W*x_last + r
        # TODO: 1.将最后一个商品的嵌入（即用户的最近兴趣表示）替换
        # feat_v = self.fc_v(feat[last_nodes])

        # 保留最后一个节点的嵌入
        last_node_emb = self.fc_i(feat[last_nodes])
        last_node_emb = dgl.broadcast_nodes(g, last_node_emb)


        feat_v = self.fc_v(intend)
        feat_v = dgl.broadcast_nodes(g, feat_v)

        # e表示每个商品与最后一个商品计算出来的重要分数
        # TODO: 2.考虑商品之间的时间差的影响
        e = self.fc_e(th.sigmoid(feat_u + feat_v + last_node_emb))

        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        # 对每个商品嵌入进行加权求和
        # TODO: 2.这里也可以考虑商品之间的时间差的影响进行加权融合,即引入另一个权重值
        feat_norm = feat * alpha

        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class Intend(nn.Module):
    def __init__(
            self, input_dim, output_dim, seq_len, batch_norm=True, feat_drop=0.0, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(seq_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, output_dim, batch_first=True)
        self.activation = activation
        self.input_dim = input_dim

    def forward(self, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        seq_len = len(feat)
        feat = self.feat_drop(feat)
        # feat = feat.view([1, seq_len, self.input_dim])
        _, intend = self.gru(feat)

        if self.activation is not None:
            intend = self.activation(intend)
        return intend.squeeze(0)


class PriceAware(nn.Module):
    def __init__(
            self, input_dim, output_dim, batch_norm=True, activation=None
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.price_factor = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.input_dim = input_dim

    def forward(self, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        price_factor = self.price_factor(feat)
        if self.activation is not None:
            price_factor = self.activation(price_factor)
        return price_factor


class LESSR(nn.Module):
    def __init__(
            self, num_items, num_category, num_price, embedding_dim, num_layers, batch_norm=True, feat_drop=0.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.category_embedding = nn.Embedding(num_category, embedding_dim, max_norm=1)
        # TODO:初始化价格嵌入矩阵
        self.price_embedding = nn.Embedding(num_price, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim

        # 设置意图GRU的输出维度: 160
        intend_output_dim = input_dim * (num_layers + 1)
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
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg, sg=None):
        device = th.device('cuda')
        # 获取包含每个序列图的list
        # 太耗时,每100个batch增加约50s
        # each_g = dgl.unbatch(mg)

        # batch_num_nodes():每张图的无重复节点数
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

        # 获取商品ID、类别ID、价格ID
        iid = mg.ndata['iid']
        iid = iid.long()

        cid = mg.ndata['cid']
        cid = cid.long()

        # TODO:获取价格ID序列
        pid = mg.ndata['pid']
        pid = pid.long()

        # 获取商品的价格嵌入
        price_embedding = self.price_embedding(pid)
        # 获取商品的类别初始嵌入
        category_feat = self.category_embedding(cid)
        # 获取商品的初始嵌入
        feat = self.embedding(iid)

        # TODO:与类别嵌入计算权重
        price_category = price_embedding.mul(category_feat)
        price_factor = self.price_aware(price_category)

        # 每个序列获取一个意图表示,将其构成一个意图矩阵[batch_size, 160]
        # 用以替代s_l,即最后一个节点的表示矩阵
        # TODO: 添: 根据类别嵌入序列获取意图表示
        fill_intend_matrix = th.zeros((20, self.embedding_dim))

        # 非unbatch()方式
        # 一个batch的类别序列矩阵
        input_cate_matrix = th.zeros((len(all_graph_origin_cid), 20, self.embedding_dim))
        for i, each_seq in enumerate(all_graph_origin_cid):
            each_seq = th.from_numpy(np.array(each_seq).astype(dtype=np.int32)).long().to(device)
            each_cate_feat = self.category_embedding(each_seq)
            # 类别序列进行零填充
            if len(each_cate_feat) < 20:
                for j in list(range(1, len(each_cate_feat) + 1))[::-1]:
                    fill_intend_matrix[-j] = each_cate_feat[-j]

        # unbatch()方式
        # input_cate_matrix = th.zeros((len(each_g), 20, 32))
        # for i, each_seq in enumerate(each_g):
        #     each_seq_cid = each_seq.ndata['origin_cid'].long()
        #     each_cate_feat = self.category_embedding(each_seq_cid)
        #     # 类别序列进行零填充
        #     if len(each_cate_feat) < 20:
        #         for j in list(range(1, len(each_cate_feat) + 1))[::-1]:
        #             fill_intend_matrix[-j] = each_cate_feat[-j]

            # 将每个填充后的类别序列堆叠成一个三维矩阵[batch_size, seq_len, dim]
            input_cate_matrix[i] = fill_intend_matrix
        input_cate_matrix = input_cate_matrix.to(device)
        intend_matrix = self.intend(input_cate_matrix)
        # [512, 160]
        intend_matrix = intend_matrix.to(device)

        # TODO: 1.添加上类别嵌入
        # feat = feat.mul(category_feat)
        # TODO: 2.添加上价格嵌入
        # feat = feat.mul(price_embedding)

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
        # TODO: 1.可以加上所有节点对应的位置嵌入表示
        # sr_g = self.readout(mg, feat, last_nodes)

        # TODO: 2.用意图表示替换最后一个节点表示,公式(13)
        # sr_g: [512, 32] 512表示batch_size
        # TODO: 用价格表示更新节点表示
        feat = price_factor.mul(feat)
        sr_g = self.readout(mg, feat, intend_matrix, last_nodes)

        # 获取最后一个节点经过多层学习后的表示,size=[batch_size, 160]
        sr_l = feat[last_nodes]
        # 获取会话整体表示,sr: [512, 192]
        # TODO: 可以考虑其它获取会话整体表示的方式,此处仅为简单的级联
        # TODO: 此处将最后一个节点表示替换成意图表示
        # sr = th.cat([sr_l, sr_g], dim=1)
        sr = th.cat([intend_matrix, sr_g], dim=1)

        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        sr = self.fc_sr(self.feat_drop(sr))

        # 利用会话表示与商品的初始嵌入计算推荐分数 [512, 42596]
        # TODO: 考虑将初始的商品嵌入进行替换 (商品嵌入+对应的类别嵌入)
        # self.indices是大小为42596的一维张量,[0,1,2,...,42594,42595],表示所有商品的索引
        logits = sr @ self.embedding(self.indices).t()
        return logits
