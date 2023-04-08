import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict


from utils import generate_binomial_mask

class MinEuclideanDistBlock(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data set and performs global min-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                               dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):
        """
        1) Unfold the data set 2) calculate euclidean distance 3) sum over channels and 4) perform global min-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the euclidean for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        
        
        
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        #x = torch.cdist(x, self.shapelets, p=2)
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        
        
        """
        n_dims = x.shape[1]
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :]
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            out += torch.cdist(x_dim, self.shapelets[i_dim : i_dim + 1, :, :], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        x = out
        x = x.transpose(2, 3)
        """
        
        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        self.shapelets.retain_grad()

        
class MaxCosineSimilarityBlock(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):
        """
        1) Unfold the data set 2) calculate norm of the data and the shapelets 3) calculate pair-wise dot-product
        4) sum over channels 5) perform a ReLU to ignore the negative values and 6) perform global max-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the cosine similarity for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        """
        n_dims = x.shape[1]
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        shapelets_norm = shapelets_norm.transpose(1, 2).half()
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :].half()
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            x_dim = x_dim / x_dim.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
            out += torch.matmul(x_dim, shapelets_norm[i_dim : i_dim + 1, :, :]).float()
        
        x = out.transpose(2, 3) / n_dims
        """
        
        
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        
       
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        
        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
       
        
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims
        
        
        # ignore negative distances
        x = self.relu(x)
        x, _ = torch.max(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)} "
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)} "
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        

class MaxCrossCorrelationBlock(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set, implemented via convolution and
    performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    # TODO Why is this multiple time slower than the other two implementations?
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.cuda()
        
        
        
    def forward(self, x, masking=False):
        """
        1) Apply 1D convolution 2) Apply global max-pooling
        @param x: the data set of time series
        @type x: array(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the most similar values for each pair of shapelet and time series instance
        @rtype: tensor(n_samples, num_shapelets)
        """
        x = self.shapelets(x)
        if masking:
            mask = generate_binomial_mask(x.shape)
            x *= mask
        x, _ = torch.max(x, 2, keepdim=True)
        return x.transpose(2, 1)

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.weight.data

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()

        if not list(weights.shape) == list(self.shapelets.weight.data.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data.shape)} compared to {list(weights.shape)}")

        self.shapelets.weight.data = weights

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets.weight.data[j, :].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data[j, :].shape)} compared to {list(weights.shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets.weight.data[j, :] = weights



class ShapeletsDistBlocks(nn.Module):
    """
    Defines a number of blocks containing a number of shapelets, whereas
    the shapelets in each block have the same size.
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets//3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda))
            self.blocks = nn.ModuleList(module_list)
        
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x, masking=False):
        """
        Calculate the distances of each shapelet block to the time series data x and concatenate the results.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: a distance matrix containing the distances of each shapelet to the time series data
        @rtype: tensor(float) of shape
        """
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        for block in self.blocks:
            if self.checkpoint and self.dist_measure != 'cross-correlation':
                out = torch.cat((out, checkpoint(block, x, masking)), dim=2)
            
            else:
                out = torch.cat((out, block(x, masking)), dim=2)
            
       

        return out

    def get_blocks(self):
        """
        @return: the list of shapelet blocks
        @rtype: nn.ModuleList
        """
        return self.blocks

    def get_block(self, i):
        """
        Get a specific shapelet block. The blocks are ordered (ascending) according to the shapelet lengths.
        @param i: the index of the block to fetch
        @type i: int
        @return: return shapelet block i
        @rtype: nn.Module, either
        """
        return self.blocks[i]

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of the shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_shapelet_weights(weights)

    def get_shapelets_of_block(self, i):
        """
        Return the shapelet of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @return: the weights of the shapelet block
        @rtype: tensor(float) of shape (in_channels, num_shapelets, shapelets_size)
        """
        return self.blocks[i].get_shapelets()

    def get_shapelet(self, i, j):
        """
        Return the shapelet at index j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @return: return the weights of the shapelet
        @rtype: tensor(float) of shape
        """
        shapelet_weights = self.blocks[i].get_shapelets()
        return shapelet_weights[j, :]

    def set_shapelet_weights_of_single_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the new weights for the shapelet
        @type weights: array-like of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_weights_of_single_shapelet(j, weights)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        max_shapelet_len = max(self.shapelets_size_and_len.keys())
        num_total_shapelets = sum(self.shapelets_size_and_len.values())
        shapelets = torch.Tensor(num_total_shapelets, self.in_channels, max_shapelet_len)
        shapelets[:] = np.nan
        start = 0
        for block in self.blocks:
            shapelets_block = block.get_shapelets()
            end = start + block.num_shapelets
            shapelets[start:end, :, :block.shapelets_size] = shapelets_block
            start += block.num_shapelets
        return shapelets

class LearningShapeletsModel(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the logits for the class predictions of the model
        @rtype: tensor(float) of shape (num_samples, num_classes)
        """
        x = self.shapelets_blocks(x, masking)
        
        x = torch.squeeze(x, 1)
        
        # test torch.cat
        #x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)
        
        x = self.projection(x)
        
        if optimize == 'acc':
            x = self.linear(x)
        
        
        return x

    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        return self.shapelets_blocks(X)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j in shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)


class LearningShapeletsModelMixDistances(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='mix',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModelMixDistances, self).__init__()

        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        
        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='euclidean', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=to_cuda, checkpoint=checkpoint)
        
        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] - 2 * (item[1] // 3) for item in shapelets_size_and_len.items()},
                                                    dist_measure='cross-correlation', to_cuda=to_cuda, checkpoint=checkpoint)
        
        
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )
        
        self.bn1 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn2 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn3 = nn.BatchNorm1d(num_features=sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values()))
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the logits for the class predictions of the model
        @rtype: tensor(float) of shape (num_samples, num_classes)
        """

        
        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)
        
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        
        x_out = self.shapelets_euclidean(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn1(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        x_out = self.shapelets_cosine(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn2(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        x_out = self.shapelets_cross_correlation(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn3(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)
        
        
        out = out.reshape(n_samples, -1)
        
        
        
        #print(out.shape)
        #out = self.projection(out)
        
        if optimize == 'acc':
            out = self.linear(out)
        
        
        return out

