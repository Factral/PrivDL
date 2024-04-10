import numpy as np
import torch
import torch.nn as nn

from .scrambler import ImageScrambler


def calculate_index_pool(dims, stride, padding, kernel_size):
    """
    Calculate the center indices for pooling operations over an input matrix.

    Args:
        dims (int): The dimension (either height or width) of the input square matrix.
        stride (int): The stride of the pooling operation.
        padding (int): The padding applied to the input matrix on each side.
        kernel_size (int): The size of the pooling kernel.

    Returns:
        np.ndarray: An array of center indices for the pooling operations.
    """
    output_dims = (dims + 2 * padding - kernel_size) // stride + 1
    center_indices = np.zeros((output_dims, output_dims, 2), dtype=int)

    indices = np.arange(- padding + kernel_size // 2 - 1, dims - padding + kernel_size // 2 - 1, stride)
    center_indices[:, :, 0], center_indices[:, :, 1] = np.meshgrid(indices, indices, indexing='ij')

    return center_indices


def get_index_pool(index, size):
    """
    Get indices of all pixels within a pooling window.

    Args:
        index (list[int, int]): The top-left corner [row, column] index of the pooling window.
        size (int): The size (dimension) of the pooling window.

    Returns:
        list[list[int, int]]: A list of [row, column] indices for each pixel within the pooling window.
    """
    return [[index[0] + i, index[1] + j] for i in range(size) for j in range(size)]


def get_pixels_from_coordinates(input_tensor, coordinates):
    """
    Extracts pixels from the input tensor at specified coordinates.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        coordinates (list[list[int, int]]): List of [row, column] coordinates to extract pixels from.

    Returns:
        torch.Tensor: Tensor of extracted pixels of shape (batch, channels, len(coordinates)).
    """
    batch_size, channels, _, _ = input_tensor.shape
    pixels = torch.zeros((batch_size, channels, len(coordinates)), dtype=input_tensor.dtype, device=input_tensor.device)

    for i, (row, col) in enumerate(coordinates):
        pixels[:, :, i] = input_tensor[:, :, row, col]

    return pixels


def retrieve_original_pixels(positions,perm):
  pixel_positions = np.array([perm.get_pos_pixel(xi) for xi in perm.key])
  pix_pos = []
  for pos in positions:
    index = np.where((pixel_positions == pos).all(axis=1))[0]
    fake_pos = perm.get_pos_pixel(index)
    pix_pos.append(fake_pos.squeeze())
  return pix_pos


class DeformMaxPool2d(nn.Module):
    """
    A deformable max pooling layer that applies a max pooling operation with the ability
    to deform based on a provided shuffling.

    Attributes:
        dim (int): Dimension of the input feature map (assumed square).
        perm (ImageScrambler): An instance of ImageScrambler for shuffling.
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling operation.
        padding (int): Padding applied to the input feature map.
    """
    def __init__(self,
                 dim,
                 perm,
                 kernel_size,
                 stride=None,
                 padding=0,
                 disorder=True) -> None:
        super(DeformMaxPool2d, self).__init__()

        self.perm = perm
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dim = dim

        if stride is None:
            self.stride = kernel_size

        self.output_dims = (dim + 2 * padding - (kernel_size - 1) - 1) // self.stride + 1
        self.new_perm = ImageScrambler(self.output_dims)

        self.index_positions = calculate_index_pool(dim, self.stride, self.padding, self.kernel_size)
        self.index_positions = np.transpose(self.index_positions, (2, 0, 1))

        if disorder:
            self.index_positions_ = self.new_perm.forward(torch.from_numpy(self.index_positions).unsqueeze(0)).squeeze(0).numpy()
        else:
            self.index_positions_ = self.index_positions

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        result = torch.zeros(batch_size, channels, self.output_dims, self.output_dims)

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                positions = get_index_pool(self.index_positions_[:, i, j], self.kernel_size)
                org_position = retrieve_original_pixels(positions, self.perm)
                pixels = get_pixels_from_coordinates(x, org_position)
                result[:, :, i, j] = torch.amax(pixels, 2)

        return result


def deform_maxPool2d(dim,perm,kernel_size,stride=None,padding=0):
    """
    functional version of the DeformMaxPool2d class
    """
    dims = dim

    if stride is None:
        stride = kernel_size

    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    new_perm = ImageScrambler(output_dims)

    index_positions = calculate_index_pool(dims,stride,padding,kernel_size)

    index_positions = np.transpose(index_positions, (2, 0, 1))
        
    index_positions_ = new_perm.forward(torch.from_numpy(index_positions).unsqueeze(0)).squeeze(0).numpy()

    batch_size, channels, _, _ = input.shape

    result = torch.zeros(batch_size,channels,output_dims,output_dims)

    for i in range(output_dims):
        for j in range(output_dims):
            positions = get_index_pool(index_positions_[:, i, j], kernel_size)
            org_position = retrieve_original_pixels(positions,perm)
            pixels = get_pixels_from_coordinates(input, org_position)
            result[:,:,i,j] = torch.amax(pixels, 2)

    return result, new_perm
