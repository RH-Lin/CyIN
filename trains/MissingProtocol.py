import numpy as np
import torch
import random
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint

class MissingProtocol:
    def __init__(self, args, device):
        """
        Initializes the MissingProtocol class based on the given arguments.
        
        Args:
            args (dict): A dictionary containing missing modality settings.
            device (str): The device to use for tensor operations (default: 'cuda').
        """
        self.args = args
        self.device = device
        self.miss_protocol = self.args['miss_protocol']

        if self.miss_protocol == 'fix':
            self.modal_type = self.args[self.miss_protocol]['modal_type']
        elif self.miss_protocol == 'random':
            self.miss_rate = self.args[self.miss_protocol].get('miss_rate', False)
            self.miss_prob = self.args[self.miss_protocol].get('miss_prob', False)
            assert bool(self.miss_rate) != bool(self.miss_prob), (
                "Only one of 'MISS_RATE'/'MISS_PROB' can have a value for missing modality evaluation. "
                "If you want complete multimodal learning, set 'MISS_PROTOCOL' as 'fix' with 'modal_type=[0,1,2]'!"
            )
        else:
            raise ValueError(f"Invalid 'MISS_PROTOCOL' setting! Choose 'fix' or 'random', not {self.miss_protocol}.")

    def get_mask_sample(self, view_num, sample_num):
        """
        Generates a mask matrix based on the selected missing protocol.

        Args:
            view_num (int): Number of modalities (views).
            sample_num (int): Number of samples.

        Returns:
            np.ndarray or torch.IntTensor: The generated mask matrix.
        """
        # Fixed Missing Modalities Protocol
        if self.miss_protocol == 'fix':
            mask_matrix = torch.zeros((sample_num, view_num), dtype=torch.int, device=self.device)
            mask_matrix[:, list(self.modal_type)] = 1.0
            return mask_matrix

        # Random Missing Modalities Protocol
        elif self.miss_protocol == 'random':

            # Dataset-level Evaluation with Missing Rate
            if self.miss_rate is not False:
                one_rate = 1 - self.miss_rate  # Ratio of modality being present

                if one_rate <= (1 / view_num):
                    # If preservation probability is very low, each sample randomly preserves one modality.
                    rand_idx = torch.randint(low=0, high=view_num, size=(sample_num,), device=self.device)
                    mask_matrix = torch.zeros((sample_num, view_num), device=self.device, dtype=torch.int32)
                    mask_matrix[torch.arange(sample_num), rand_idx] = 1

                else:
                    error = 1.0
                    test_time = 0
                    error_threshold = 0.005
                    # Iteratively adjust until the overall preserved ratio is within 0.005 of one_rate.
                    while error >= error_threshold:
                        # scale error threshold to prevent always circling
                        test_time+=1
                        if test_time > 100:
                            error_threshold = error_threshold * 2
                            print(f"Note: Turning ERROR_THRESHOLD in MissingProtocol into {error_threshold} , pls reset the testing BATCH_SIZE into a larger value to fit in the unified evaluation setting !!!")
                            test_time = 0

                        # For each sample, randomly choose one modality to definitely preserve.
                        rand_idx = torch.randint(low=0, high=view_num, size=(sample_num,), device=self.device)
                        view_preserve = torch.zeros((sample_num, view_num), device=self.device, dtype=torch.int32)
                        view_preserve[torch.arange(sample_num), rand_idx] = 1

                        # Calculate the total number of additional ones needed
                        one_num = view_num * sample_num * one_rate - sample_num
                        ratio = one_num / (view_num * sample_num)
                        matrix_iter = (torch.randint(0, 100, (sample_num, view_num), device=self.device) < int(ratio * 100)).to(torch.int32)
                        combined = matrix_iter + view_preserve
                        # Count the overlapping ones (positions where combined value > 1)
                        a = torch.sum((combined > 1).to(torch.int32)).item()
                        if a / one_num == 1:
                            one_num_iter = one_num
                        else:
                            one_num_iter = one_num / (1 - a / one_num)
                        ratio = one_num_iter / (view_num * sample_num)
                        matrix_iter = (torch.randint(0, 100, (sample_num, view_num), device=self.device) < int(ratio * 100)).to(torch.int32)
                        mask_matrix = ((matrix_iter + view_preserve) > 0).to(torch.int32)
                        current_ratio = torch.sum(mask_matrix).item() / (view_num * sample_num)
                        error = abs(one_rate - current_ratio)

            # Instance-level Evaluation with Missing Probability
            if self.miss_prob is not False:
                miss_patterns = torch.tensor([
                    [1, 0, 1], [1, 1, 0], [0, 1, 1],
                    [1, 0, 0], [0, 0, 1], [0, 1, 0]
                ], dtype=torch.int, device=self.device)

                # Default mask: all modalities are visible
                mask_matrix = torch.ones((sample_num, view_num), dtype=torch.int, device=self.device)

                # Assign a missing pattern to each sample
                miss_random = miss_patterns[torch.randint(0, miss_patterns.shape[0], (sample_num,), device=self.device)]

                # Bernoulli sampling to determine which samples should have missing modalities
                apply_mask = torch.rand(sample_num, device=self.device) < self.miss_prob
                mask_matrix[apply_mask] = miss_random[apply_mask]
                
        return mask_matrix.to(self.device)
