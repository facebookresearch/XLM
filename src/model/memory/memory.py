from logging import getLogger
import math
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...utils import bool_flag
from .utils import get_knn_faiss, cartesian_product
from .utils import get_gaussian_keys, get_uniform_keys
from .query import QueryIdentity, QueryMLP, QueryConv


logger = getLogger()


class HashingMemory(nn.Module):

    MEM_VALUES_PARAMS = '.values.weight'
    VALUES = None
    EVAL_MEMORY = True
    _ids = itertools.count(0)

    def __init__(self, input_dim, output_dim, params):

        super().__init__()
        self.id = next(self._ids)

        # global parameters
        self.input2d = params.mem_input2d
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = params.mem_size
        self.modulo_size = params.mem_modulo_size
        self.n_indices = params.n_indices
        self.k_dim = params.mem_k_dim
        self.v_dim = params.mem_v_dim if params.mem_v_dim > 0 else output_dim
        self.heads = params.mem_heads
        self.knn = params.mem_knn
        self.shuffle_indices = params.mem_shuffle_indices
        self.keys_normalized_init = params.mem_keys_normalized_init
        self.product_quantization = params.mem_product_quantization
        assert self.modulo_size == -1 and self.size == self.n_indices or self.n_indices > self.size == self.modulo_size >= 1

        # keys / queries
        self.keys_type = params.mem_keys_type
        self.learn_keys = params.mem_keys_learn
        self.use_different_keys = params.mem_use_different_keys
        self.query_detach_input = params.mem_query_detach_input
        self.query_net_learn = params.mem_query_net_learn
        self.multi_query_net = params.mem_multi_query_net
        self.shuffle_query = params.mem_shuffle_query
        assert self.use_different_keys is False or self.keys_type in ['gaussian', 'uniform']
        assert self.use_different_keys is False or self.heads >= 2 or self.product_quantization
        assert self.multi_query_net is False or self.heads >= 2 or self.product_quantization
        assert self.shuffle_query is False or self.heads > 1 and params.mem_query_layer_sizes == ''
        assert self.shuffle_query is False or self.input_dim % (2 ** self.heads) == 0

        # scoring / re-scoring
        self.normalize_query = params.mem_normalize_query
        self.temperature = params.mem_temperature
        self.score_softmax = params.mem_score_softmax
        self.score_subtract = params.mem_score_subtract
        self.score_normalize = params.mem_score_normalize
        assert self.score_subtract in ['', 'min', 'mean', 'median']
        assert self.score_subtract == '' or self.knn >= 2
        assert not (self.score_normalize and self.score_softmax and self.score_subtract == '')

        # dropout
        self.input_dropout = params.mem_input_dropout
        self.query_dropout = params.mem_query_dropout
        self.value_dropout = params.mem_value_dropout

        # initialize keys
        self.init_keys()

        # self.values = nn.Embedding(self.size, self.v_dim, sparse=params.mem_sparse)
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=params.mem_sparse)

        # optionally use the same values for all memories
        if params.mem_share_values:
            if HashingMemory.VALUES is None:
                HashingMemory.VALUES = self.values.weight
            else:
                self.values.weight = HashingMemory.VALUES

        # values initialization
        if params.mem_value_zero_init:
            nn.init.zeros_(self.values.weight)
        else:
            nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)

        # no query network
        if len(params.mem_query_layer_sizes) == 0:
            assert self.heads == 1 or self.use_different_keys or self.shuffle_query
            assert self.input_dim == self.k_dim
            self.query_proj = QueryIdentity(self.input_dim, self.heads, self.shuffle_query)

        # query network
        if len(params.mem_query_layer_sizes) > 0:
            assert not self.shuffle_query

            # layer sizes / number of features
            l_sizes = list(params.mem_query_layer_sizes)
            assert len(l_sizes) >= 2 and l_sizes[0] == l_sizes[-1] == 0
            l_sizes[0] = self.input_dim
            l_sizes[-1] = (self.k_dim // 2) if self.multi_query_net else (self.heads * self.k_dim)

            # convolutional or feedforward
            if self.input2d:
                self.query_proj = QueryConv(
                    self.input_dim, self.heads, self.k_dim, self.product_quantization,
                    self.multi_query_net, l_sizes, params.mem_query_kernel_sizes,
                    bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm,
                    grouped_conv=params.mem_grouped_conv
                )
            else:
                assert params.mem_query_kernel_sizes == ''
                assert not params.mem_query_residual
                self.query_proj = QueryMLP(
                    self.input_dim, self.heads, self.k_dim, self.product_quantization,
                    self.multi_query_net, l_sizes,
                    bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm,
                    grouped_conv=params.mem_grouped_conv
                )

        # shuffle indices for different heads
        if self.shuffle_indices:
            head_permutations = [torch.randperm(self.n_indices).unsqueeze(0) for i in range(self.heads)]
            self.register_buffer('head_permutations', torch.cat(head_permutations, 0))

        # do not learn the query network
        if self.query_net_learn is False:
            for p in self.query_proj.parameters():
                p.requires_grad = False

    def forward(self, input):
        """
        Read from the memory.
        """
        # detach input
        if self.query_detach_input:
            input = input.detach()

        # input dimensions
        if self.input2d:
            assert input.shape[1] == self.input_dim
            n_images, _, height, width = input.shape
            prefix_shape = (n_images, width, height)
        else:
            assert input.shape[-1] == self.input_dim
            prefix_shape = input.shape[:-1]

        # compute query / store it
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)    # input shape
        query = self.query_proj(input)                                            # (bs * heads, k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)    # (bs * heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # get indices
        scores, indices = self.get_indices(query, self.knn)                       # (bs * heads, knn) ** 2

        # optionally shuffle indices for different heads
        if self.shuffle_indices:
            indices = indices.view(bs, self.heads, -1).chunk(self.heads, 1)
            indices = [p[idx] for p, idx in zip(self.head_permutations, indices)]
            indices = torch.cat(indices, 1).view(bs * self.heads, -1)

        # take indices modulo the memory size
        if self.modulo_size != -1:
            indices = indices % self.modulo_size

        # re-scoring
        if self.temperature != 1:
            scores = scores / self.temperature                                    # (bs * heads, knn)
        if self.score_softmax:
            scores = F.softmax(scores.float(), dim=-1).type_as(scores)            # (bs * heads, knn)
        if self.score_subtract != '':
            if self.score_subtract == 'min':
                to_sub = scores.min(1, keepdim=True)[0]                           # (bs * heads, 1)
            if self.score_subtract == 'mean':
                to_sub = scores.mean(1, keepdim=True)                             # (bs * heads, 1)
            if self.score_subtract == 'median':
                to_sub = scores.median(1, keepdim=True)[0]                        # (bs * heads, 1)
            scores = scores - to_sub                                              # (bs * heads, knn)
        if self.score_normalize:
            scores = scores / scores.norm(p=1, dim=1, keepdim=True)               # (bs * heads, knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)                         # (bs, heads * knn)
        scores = scores.view(bs, self.heads * self.knn)                           # (bs, heads * knn)

        # weighted sum of values
        # output = self.values(indices) * scores.unsqueeze(-1)                    # (bs * heads, knn, v_dim)
        # output = output.sum(1)                                                  # (bs * heads, v_dim)
        output = self.values(
            indices,
            per_sample_weights=scores.to(self.values.weight.data)
        ).to(scores)                                                              # (bs, v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)  # (bs, v_dim)

        # reshape output
        if self.input2d:
            output = output.view(n_images, width, height, self.v_dim)             # (n_images, width, height, v_dim)
            output = output.transpose(1, 3)                                       # (n_images, v_dim, height, width)
        else:
            if len(prefix_shape) >= 2:
                output = output.view(prefix_shape + (self.v_dim,))                # (..., v_dim)

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, self.knn).detach().cpu().float()

        return output

    def init_keys(self):
        raise Exception("Not implemented!")

    def _get_indices(self, query, knn, keys):
        raise Exception("Not implemented!")

    def get_indices(self, query, knn):
        raise Exception("Not implemented!")

    @staticmethod
    def register_args(parser):
        """
        Register memory parameters
        """
        # memory implementation
        parser.add_argument("--mem_implementation", type=str, default="pq_fast",
                            help="Memory implementation (flat, pq_default, pq_fast)")

        # optimization
        parser.add_argument("--mem_grouped_conv", type=bool_flag, default=False,
                            help="Use grouped convolutions in the query network")
        parser.add_argument("--mem_values_optimizer", type=str, default="adam,lr=0.001",
                            help="Memory values optimizer ("" for the same optimizer as the rest of the model)")
        parser.add_argument("--mem_sparse", type=bool_flag, default=False,
                            help="Perform sparse updates for the values")

        # global parameters
        parser.add_argument("--mem_input2d", type=bool_flag, default=False,
                            help="Convolutional query network")
        parser.add_argument("--mem_k_dim", type=int, default=256,
                            help="Memory keys dimension")
        parser.add_argument("--mem_v_dim", type=int, default=-1,
                            help="Memory values dimension (-1 for automatic output dimension)")
        parser.add_argument("--mem_heads", type=int, default=4,
                            help="Number of memory reading heads")
        parser.add_argument("--mem_knn", type=int, default=32,
                            help="Number of memory slots to read / update - k-NN to the query")
        parser.add_argument("--mem_share_values", type=bool_flag, default=False,
                            help="Share values across memories")
        parser.add_argument("--mem_shuffle_indices", type=bool_flag, default=False,
                            help="Shuffle indices for different heads")
        parser.add_argument("--mem_shuffle_query", type=bool_flag, default=False,
                            help="Shuffle query dimensions (when the query network is the identity and there are multiple heads)")
        parser.add_argument("--mem_modulo_size", type=int, default=-1,
                            help="Effective memory size: indices are taken modulo this parameter. -1 to disable.")

        # keys
        parser.add_argument("--mem_keys_type", type=str, default="uniform",
                            help="Memory keys type (binary,gaussian,uniform)")
        parser.add_argument("--mem_n_keys", type=int, default=512,
                            help="Number of keys")
        parser.add_argument("--mem_keys_normalized_init", type=bool_flag, default=False,
                            help="Normalize keys at initialization")
        parser.add_argument("--mem_keys_learn", type=bool_flag, default=True,
                            help="Learn keys")
        parser.add_argument("--mem_use_different_keys", type=bool_flag, default=True,
                            help="Use different keys for each head / product quantization")

        # queries
        parser.add_argument("--mem_query_detach_input", type=bool_flag, default=False,
                            help="Detach input")
        parser.add_argument("--mem_query_layer_sizes", type=str, default="0,0",
                            help="Query MLP layer sizes ('', '0,0', '0,512,0')")
        parser.add_argument("--mem_query_kernel_sizes", type=str, default="",
                            help="Query MLP kernel sizes (2D inputs only)")
        parser.add_argument("--mem_query_bias", type=bool_flag, default=True,
                            help="Query MLP bias")
        parser.add_argument("--mem_query_batchnorm", type=bool_flag, default=False,
                            help="Query MLP batch norm")
        parser.add_argument("--mem_query_net_learn", type=bool_flag, default=True,
                            help="Query MLP learn")
        parser.add_argument("--mem_query_residual", type=bool_flag, default=False,
                            help="Use a bottleneck with a residual layer in the query MLP")
        parser.add_argument("--mem_multi_query_net", type=bool_flag, default=False,
                            help="Use multiple query MLP (one for each head)")

        # values initialization
        parser.add_argument("--mem_value_zero_init", type=bool_flag, default=False,
                            help="Initialize values with zeros")

        # scoring
        parser.add_argument("--mem_normalize_query", type=bool_flag, default=False,
                            help="Normalize queries")
        parser.add_argument("--mem_temperature", type=float, default=1,
                            help="Divide scores by a temperature")
        parser.add_argument("--mem_score_softmax", type=bool_flag, default=True,
                            help="Apply softmax on scores")
        parser.add_argument("--mem_score_subtract", type=str, default="",
                            help="Subtract scores ('', min, mean, median)")
        parser.add_argument("--mem_score_normalize", type=bool_flag, default=False,
                            help="L1 normalization of the scores")

        # dropout
        parser.add_argument("--mem_input_dropout", type=float, default=0,
                            help="Input dropout")
        parser.add_argument("--mem_query_dropout", type=float, default=0,
                            help="Query dropout")
        parser.add_argument("--mem_value_dropout", type=float, default=0,
                            help="Value dropout")

    @staticmethod
    def build(input_dim, output_dim, params):
        if params.mem_implementation == 'flat':
            M = HashingMemoryFlat
        elif params.mem_implementation == 'pq_default':
            M = HashingMemoryProduct
        elif params.mem_implementation == 'pq_fast':
            M = HashingMemoryProductFast
        else:
            raise Exception("Unknown memory implementation!")
        return M(input_dim, output_dim, params)

    @staticmethod
    def check_params(params):
        """
        Check and initialize memory parameters.
        """
        # memory
        assert params.mem_implementation in ['flat', 'pq_default', 'pq_fast']
        params.mem_product_quantization = params.mem_implementation != 'flat'

        # optimization
        assert params.mem_grouped_conv is False or params.mem_multi_query_net
        params.mem_values_optimizer = params.optimizer if params.mem_values_optimizer == '' else params.mem_values_optimizer
        params.mem_values_optimizer = params.mem_values_optimizer.replace('adam', 'sparseadam') if params.mem_sparse else params.mem_values_optimizer

        # even number of key dimensions for product quantization
        assert params.mem_k_dim >= 2
        assert params.mem_product_quantization is False or params.mem_k_dim % 2 == 0

        # memory type
        assert params.mem_keys_type in ['binary', 'gaussian', 'uniform']

        # number of indices
        if params.mem_keys_type == 'binary':
            assert params.mem_keys_normalized_init is False
            assert 1 << params.mem_k_dim == params.mem_n_keys
        if params.mem_product_quantization:
            params.n_indices = params.mem_n_keys ** 2
        else:
            params.n_indices = params.mem_n_keys

        # actual memory size
        if params.mem_modulo_size == -1:
            params.mem_size = params.n_indices
        else:
            assert 1 <= params.mem_modulo_size < params.n_indices
            params.mem_size = params.mem_modulo_size

        # different keys / different query MLP / shuffle hidden dimensions when no query network
        assert not params.mem_use_different_keys or params.mem_keys_type in ['gaussian', 'uniform']
        assert not params.mem_use_different_keys or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_query_layer_sizes not in ['', '0,0']
        assert not params.mem_shuffle_query or params.mem_heads > 1 and params.mem_query_layer_sizes == ''

        # query network
        if params.mem_query_layer_sizes == '':
            assert params.mem_heads == 1 or params.mem_use_different_keys or params.mem_shuffle_query
        else:
            s = [int(x) for x in filter(None, params.mem_query_layer_sizes.split(','))]
            assert len(s) >= 2 and s[0] == s[-1] == 0
            params.mem_query_layer_sizes = s
            assert not params.mem_query_residual or params.mem_input2d

        # convolutional query network kernel sizes
        if params.mem_query_kernel_sizes == '':
            assert not params.mem_input2d or params.mem_query_layer_sizes == ''
        else:
            assert params.mem_input2d
            s = [int(x) for x in filter(None, params.mem_query_kernel_sizes.split(','))]
            params.mem_query_kernel_sizes = s
            assert all(ks % 2 == 1 for ks in s)
            assert len(params.mem_query_kernel_sizes) == len(params.mem_query_layer_sizes) - 1 >= 1

        # scoring
        assert params.mem_score_subtract in ['', 'min', 'mean', 'median']
        assert params.mem_score_subtract == '' or params.mem_knn >= 2
        assert not (params.mem_score_normalize and params.mem_score_softmax and params.mem_score_subtract == '')

        # dropout
        assert 0 <= params.mem_input_dropout < 1
        assert 0 <= params.mem_query_dropout < 1
        assert 0 <= params.mem_value_dropout < 1

        # query batchnorm
        if params.mem_query_batchnorm:
            logger.warning("WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer.")


class HashingMemoryFlat(HashingMemory):

    def __init__(self, input_dim, output_dim, params):
        super().__init__(input_dim, output_dim, params)
        assert self.use_different_keys is False or self.heads >= 2
        assert not self.product_quantization

    def init_keys(self):
        """
        Initialize keys.
        """
        assert self.keys_type in ['binary', 'gaussian', 'uniform']

        # binary keys
        if self.keys_type == 'binary':
            keys = torch.FloatTensor(2 ** self.k_dim, self.k_dim)
            for i in range(keys.shape[0]):
                for j in range(keys.shape[1]):
                    keys[i, j] = int((1 << j) & i > 0)
            keys *= 2
            keys -= 1
            keys /= math.sqrt(self.k_dim)

        # random keys from Gaussian or uniform distributions
        if self.keys_type in ['gaussian', 'uniform']:
            init = get_gaussian_keys if self.keys_type == 'gaussian' else get_uniform_keys
            if self.use_different_keys:
                keys = torch.from_numpy(np.array([
                    init(self.n_indices, self.k_dim, self.keys_normalized_init, seed=i)
                    for i in range(self.heads)
                ])).view(self.heads, self.n_indices, self.k_dim)
            else:
                keys = torch.from_numpy(init(self.n_indices, self.k_dim, self.keys_normalized_init, seed=0))

        # learned or fixed keys
        if self.learn_keys:
            self.keys = nn.Parameter(keys)
        else:
            self.register_buffer('keys', keys)

    # def _get_indices(self, query, knn, keys):
    #     """
    #     Generate scores and indices given keys and unnormalized queries.
    #     """
    #     QUERY_SIZE = 4096
    #     assert query.dim() == 2 and query.size(1) == self.k_dim

    #     # optionally normalize queries
    #     if self.normalize_query:
    #         query = query / query.norm(2, 1, keepdim=True).expand_as(query)  # (bs, kdim)

    #     # compute memory indices, and split the query if it is too large
    #     with torch.no_grad():
    #         if len(query) <= QUERY_SIZE:
    #             indices = get_knn_faiss(keys.float(), query.float(), knn, distance='dot_product')[1]
    #         else:
    #             indices = torch.cat([
    #                 get_knn_faiss(keys.float(), query[i:i + QUERY_SIZE].float(), knn, distance='dot_product')[1]
    #                 for i in range(0, len(query), QUERY_SIZE)
    #             ], 0)
    #             # indices0 = get_knn_faiss(keys.float(), query.float(), knn, distance='dot_product')[1]
    #             # assert (indices0 - indices).abs().sum().item() == 0
    #         assert len(indices) == len(query)

    #     # compute value scores
    #     scores = (keys[indices] * query.unsqueeze(1)).sum(2)

    #     # return scores with indices
    #     assert scores.shape == indices.shape == (query.shape[0], knn)
    #     return scores, indices

    def _get_indices(self, query, knn, keys):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim

        # optionally normalize queries
        if self.normalize_query:
            query = query / query.norm(2, 1, keepdim=True).expand_as(query)   # (bs, kdim)

        # compute scores with indices
        scores = F.linear(query, keys, bias=None)                             # (bs, n_keys)
        scores, indices = scores.topk(knn, dim=1, largest=True, sorted=True)  # (bs, knn) ** 2
        # scores, indices = get_knn_faiss(keys.float(), query.float().contiguous(), knn, distance='dot_product')   # (bs, knn) ** 2

        # return scores with indices
        assert scores.shape == indices.shape == (query.shape[0], knn)
        return scores, indices

    def get_indices(self, query, knn):
        """
        Generate scores and indices given unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        if self.use_different_keys is False:
            return self._get_indices(query, knn, self.keys)
        else:
            bs = len(query)
            query = query.view(-1, self.heads, self.k_dim)
            outputs = [
                self._get_indices(query[:, i], knn, self.keys[i])
                for i in range(self.heads)
            ]
            scores = torch.cat([s.unsqueeze(1) for s, _ in outputs], 1).view(bs, knn)
            indices = torch.cat([idx.unsqueeze(1) for _, idx in outputs], 1).view(bs, knn)
            return scores, indices


class HashingMemoryProduct(HashingMemory):

    def __init__(self, input_dim, output_dim, params):
        super().__init__(input_dim, output_dim, params)
        assert self.k_dim % 2 == 0
        assert self.product_quantization

    def create_keys(self):
        """
        This function creates keys and returns them.
        I guess you could see that from the name of the function and the fact that is has a return statement.
        """
        assert self.keys_type in ['binary', 'gaussian', 'uniform']
        half = self.k_dim // 2
        n_keys = int(self.n_indices ** 0.5)

        # binary keys
        if self.keys_type == 'binary':
            keys = torch.FloatTensor(2 ** half, half)
            for i in range(keys.shape[0]):
                for j in range(keys.shape[1]):
                    keys[i, j] = int((1 << j) & i > 0)
            keys *= 2
            keys -= 1
            keys /= math.sqrt(self.k_dim)

        # random keys from Gaussian or uniform distributions
        if self.keys_type in ['gaussian', 'uniform']:
            init = get_gaussian_keys if self.keys_type == 'gaussian' else get_uniform_keys
            if self.use_different_keys:
                keys = torch.from_numpy(np.array([
                    init(n_keys, half, self.keys_normalized_init, seed=(2 * i + j))
                    for i in range(self.heads)
                    for j in range(2)
                ])).view(self.heads, 2, n_keys, half)
            else:
                keys = torch.from_numpy(init(n_keys, half, self.keys_normalized_init, seed=0))

        return keys

    def init_keys(self):
        """
        Initialize keys.
        """
        keys = self.create_keys()

        # learned or fixed keys
        if self.learn_keys:
            self.keys = nn.Parameter(keys)
        else:
            self.register_buffer('keys', keys)

    def _get_indices(self, query, knn, keys1, keys2):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        assert len(keys1) == len(keys2)
        half = self.k_dim // 2
        n_keys = len(keys1)

        # split query for product quantization
        q1 = query[:, :half]                                                                            # (bs, half)
        q2 = query[:, half:]                                                                            # (bs, half)

        # optionally normalize queries
        if self.normalize_query:
            q1 = q1 / q1.norm(2, 1, keepdim=True).expand_as(q1)                                         # (bs, half)
            q2 = q2 / q2.norm(2, 1, keepdim=True).expand_as(q2)                                         # (bs, half)

        # compute memory value indices
        with torch.no_grad():

            # compute indices with associated scores
            scores1, indices1 = get_knn_faiss(keys1.float(), q1.float(), knn, distance='dot_product')  # (bs, knn) ** 2
            scores2, indices2 = get_knn_faiss(keys2.float(), q2.float(), knn, distance='dot_product')  # (bs, knn) ** 2

            # cartesian product on best candidate keys
            concat_scores = cartesian_product(scores1, scores2)                                         # (bs, knn ** 2, 2)
            concat_indices = cartesian_product(indices1, indices2)                                      # (bs, knn ** 2, 2)

            all_scores = concat_scores.sum(2)                                                           # (bs, knn ** 2)
            all_indices = concat_indices[:, :, 0] * n_keys + concat_indices[:, :, 1]                    # (bs, knn ** 2)

            _scores, best_indices = torch.topk(all_scores, k=knn, dim=1, largest=True, sorted=True)     # (bs, knn)
            indices = all_indices.gather(1, best_indices)                                               # (bs, knn)

        # compute value scores - for some reason, this part is extremely slow when the keys are learned
        indices1 = indices / n_keys
        indices2 = indices % n_keys
        scores1 = (keys1[indices1] * q1.unsqueeze(1)).sum(2)
        scores2 = (keys2[indices2] * q2.unsqueeze(1)).sum(2)
        scores = scores1 + scores2

        # return scores with indices
        assert scores.shape == indices.shape == (query.shape[0], knn)
        return scores, indices

    def get_indices(self, query, knn):
        """
        Generate scores and indices given unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        if self.use_different_keys is False:
            return self._get_indices(query, knn, self.keys, self.keys)
        else:
            bs = len(query)
            query = query.view(-1, self.heads, self.k_dim)
            outputs = [
                self._get_indices(query[:, i], knn, self.keys[i][0], self.keys[i][1])
                for i in range(self.heads)
            ]
            scores = torch.cat([s.unsqueeze(1) for s, _ in outputs], 1).view(bs, knn)
            indices = torch.cat([idx.unsqueeze(1) for _, idx in outputs], 1).view(bs, knn)
            return scores, indices


class HashingMemoryProductFast(HashingMemoryProduct):

    def __init__(self, input_dim, output_dim, params):
        super().__init__(input_dim, output_dim, params)

    def _get_indices(self, query, knn, keys1, keys2):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        assert len(keys1) == len(keys2)
        bs = query.size(0)
        half = self.k_dim // 2
        n_keys = len(keys1)

        # split query for product quantization
        q1 = query[:, :half]                                                                                          # (bs, half)
        q2 = query[:, half:]                                                                                          # (bs, half)

        # optionally normalize queries
        if self.normalize_query:
            q1 = q1 / q1.norm(2, 1, keepdim=True).expand_as(q1)                                                       # (bs, half)
            q2 = q2 / q2.norm(2, 1, keepdim=True).expand_as(q2)                                                       # (bs, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, keys1, bias=None)                                                                      # (bs, n_keys ** 0.5)
        scores2 = F.linear(q2, keys2, bias=None)                                                                      # (bs, n_keys ** 0.5)
        scores1, indices1 = scores1.topk(knn, dim=1, largest=True, sorted=True)                                       # (bs, knn) ** 2
        scores2, indices2 = scores2.topk(knn, dim=1, largest=True, sorted=True)                                       # (bs, knn) ** 2
        # scores1, indices1 = get_knn_faiss(keys1, q1.contiguous(), knn, distance='dot_product')                        # (bs, knn) ** 2
        # scores2, indices2 = get_knn_faiss(keys2, q2.contiguous(), knn, distance='dot_product')                        # (bs, knn) ** 2

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1, largest=True, sorted=True)                        # (bs, knn)
        indices = all_indices.gather(1, best_indices)                                                                 # (bs, knn)

        # code below: debug instant retrieval speed
        # scores = torch.zeros(bs, knn, dtype=query.dtype, device=query.device)
        # indices = torch.arange(knn, dtype=torch.int64, device=query.device).view(1, knn).expand(bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices
