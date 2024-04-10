import numpy as np
import torch
from .scrambler import ImageScrambler


def get_index_conv(center, size):
    """
    Calculates the indices of all pixels within a square convolution window centered at a given pixel.

    Args:
        center (list[int]): The row and column index of the center pixel, e.g., [5, 5].
        size (int): The height and width of the square convolution window.

    Returns:
        list[list[int]]: A list of [row, column] indices for each pixel within the convolution window.
    """
    index_range = range(-(size // 2), size // 2 + 1)
    return [[center[0] + i, center[1] + j] for i in index_range for j in index_range]


def calculate_index_conv(dims, stride, padding, kernel_size):
    """
    Calculates the center indices of convolution operations over an input matrix, considering given dimensions and convolution parameters.

    Args:
        dims (int): The dimension (height/width) of the input square matrix.
        stride (int): The stride of the convolution operation.
        padding (int): The padding applied to the input matrix.
        kernel_size (int): The size of the convolution kernel.

    Returns:
        np.ndarray: An array containing the [row, column] indices of the centers of convolution operations.
    """
    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    center_indices = torch.zeros((output_dims, output_dims, 2), dtype=int)

    row_indices = torch.arange(-padding + kernel_size // 2, dims + padding - kernel_size // 2, stride)
    center_indices[:, :, 0], center_indices[:, :, 1] = torch.meshgrid(row_indices, row_indices, indexing='ij')

    return center_indices.permute(2, 0, 1) # [c,h,w]


def calculate_offset(dims, stride, padding, kernel_size, scrambler, batch=64):    
    """
    Calculates the offset for scrambling the convolution operations in a deformable convolution layer.

    Args:
        dims (int): The dimension (height/width) of the input square matrix.
        stride (int): The stride of the convolution operation.
        padding (int): The padding applied to the input matrix.
        kernel_size (int): The size of the convolution kernel.
        scrambler (ImageScrambler): An instance of ImageScrambler used for the permutation of convolution operations.
        batch (int, optional): The batch size for the operation. Defaults to 64.

    Returns:
        torch.Tensor: A tensor containing the calculated offsets for each convolution window.
        ImageScrambler: The new ImageScrambler generated for manntain the security
    """
    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    # generate key for the offset of the conv
    newScrambler = ImageScrambler(output_dims)

    center_indices = calculate_index_conv(dims, stride, padding, kernel_size)
    # shuffle conv indices
    scrambled_indices  = newScrambler.forward(center_indices.unsqueeze(0)).squeeze(0)

    pixel_positions = np.array([scrambler.get_pos_pixel(xi) for xi in scrambler.key])

    # Initialize offset array
    offset = np.zeros((2 * kernel_size ** 2, output_dims, output_dims), dtype='float32')

    for i in range(output_dims):
        for j in range(output_dims):
            scrambled_positions = get_index_conv(scrambled_indices[:, i, j], kernel_size)
            original_positions  = get_index_conv(center_indices[:, i, j], kernel_size)

            for a, pos in enumerate(scrambled_positions):
                index = np.where((pixel_positions == pos).all(axis=1))[0]
                if index.size > 0:
                    fake_pos = scrambler.get_pos_pixel(index)
                    distance = np.squeeze(fake_pos) - original_positions[a]
                else:
                    distance = np.array([0, 0]) - original_positions[a] - 1

                offset[a * 2, i, j] = distance[0]
                offset[a * 2 + 1, i, j] = distance[1]

    offset = np.tile(offset, [batch, 1, 1, 1])
    return torch.from_numpy(offset).float(), newScrambler