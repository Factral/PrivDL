import numpy as np
import torch

class ImageScrambler:
    """
    A class for scrambling and unscrambling images based on a random key.

    This class generates a random key used to scramble and subsequently unscramble images,
    ensuring that the original image can be retrieved after scrambling. The scrambling
    and unscrambling processes are implemented in the forward and backward methods, respectively.

    Attributes:
        image_size (int): The height and width of the square image to be scrambled.
        key (np.ndarray): An array of indices representing the scrambled positions of pixels.
        forward_map (torch.Tensor): A precomputed tensor for fast forward scrambling.
        backward_map (torch.Tensor): A precomputed tensor for fast backward unscrambling.
    """

    def __init__(self, image_size):
        """
        Initializes the ImageScrambler with a specified image size.

        Args:
            image_size (int): The height and width of the square image to be scrambled.
        """
        self.image_size = image_size
        self.key = np.argsort(np.random.rand(image_size ** 2))

        # Precompute forward and backward maps for efficient scrambling and unscrambling.
        self.forward_map, self.backward_map = self._precompute_maps()

    def _precompute_maps(self):
        """
        Precomputes the forward and backward mapping tensors for pixel scrambling.

        The method calculates the row and column indices for the scrambled positions
        and creates mappings for efficiently scrambling and unscrambling images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensors representing the forward and backward maps.
        """
        # Calculate 2D positions for each index in the flat key array
        scrambled_positions = np.array([self.get_pos_pixel(i) for i in self.key])
        
        # Forward map: Original to scrambled
        # We map the linear indices to their new positions directly
        forward_map = torch.LongTensor(scrambled_positions.T)  # Transpose to get two arrays for rows and columns

        # Backward map: Scrambled to original
        # Create an inverse mapping
        # For each position in the scrambled image, find its original position
        original_positions = np.argsort(self.key)
        backward_map = torch.LongTensor(np.array([self.get_pos_pixel(i) for i in original_positions]).T)

        return forward_map, backward_map

    def get_pos_pixel(self, pos):
        """
        Computes the row and column of a pixel based on its linear position.

        Args:
            pos (int): The linear position of the pixel.

        Returns:
            np.ndarray: An array containing the row and column of the pixel.
        """
        n_row = pos // self.image_size
        n_col = pos % self.image_size
        return np.array([n_row, n_col])

    def forward(self, pic):
        """
        Scrambles an image based on the precomputed key.

        Args:
            pic (torch.Tensor): The image tensor to be scrambled.

        Returns:
            torch.Tensor: The scrambled image tensor.
        """
        return pic[:, :, self.forward_map[0], self.forward_map[1]].view(pic.size())

    def backward(self, pic):
        """
        Unscrambles an image based on the precomputed key.

        Args:
            pic (torch.Tensor): The scrambled image tensor to be unscrambled.

        Returns:
            torch.Tensor: The unscrambled image tensor.
        """
        return pic[:, :, self.backward_map[0], self.backward_map[1]].view(pic.size())
