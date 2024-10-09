# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15  # longer timeout since permute calls transpose recursively

parameters = {
    "traces": {
        "permute_specs": [
            {"shape": [1, 1, 1, 7, 7, 1024], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 1, 7, 7, 768], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 1, 8, 8, 1024], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 1, 8, 8, 768], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1, 12], "dims": [2, 0, 1]},
            {"shape": [1, 1, 16, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1, 16384, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1, 16], "dims": [2, 0, 1]},
            {"shape": [1, 1, 19200, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1, 6], "dims": [2, 0, 1]},
            {"shape": [1, 1, 7, 1, 7, 1024], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 7, 1, 7, 768], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 8, 1, 8, 1024], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 8, 1, 8, 768], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 1, 8], "dims": [2, 0, 1]},
            {"shape": [1, 10, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1024, 160], "dims": [0, 2, 1]},
            {"shape": [1, 1024, 196], "dims": [0, 2, 1]},
            {"shape": [1, 1024, 256], "dims": [0, 2, 1]},
            {"shape": [1, 1024, 49], "dims": [0, 2, 1]},
            {"shape": [1, 1024, 5, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 1, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 10, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 197, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 197, 64], "dims": [2, 0, 1, 3]},
            {"shape": [1, 12, 201, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 25, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 45, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 50, 64], "dims": [2, 0, 1, 3]},
            {"shape": [1, 12, 64, 8], "dims": [0, 1, 3, 2]},
            {"shape": [1, 12, 7, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 12, 8, 64], "dims": [0, 1, 3, 2]},
            {"shape": [1, 120, 160, 64], "dims": [0, 3, 1, 2]},
            {"shape": [1, 1200, 320], "dims": [0, 2, 1]},
            {"shape": [1, 1200, 5, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 128, 128, 32], "dims": [0, 3, 1, 2]},
            {"shape": [1, 128, 300], "dims": [0, 2, 1]},
            {"shape": [1, 128, 56, 56], "dims": [0, 2, 3, 1]},
            {"shape": [1, 128, 64, 64], "dims": [0, 2, 3, 1]},
            {"shape": [1, 1280, 1369], "dims": [0, 2, 1]},
            {"shape": [1, 1280, 16, 16], "dims": [0, 2, 3, 1]},
            {"shape": [1, 1280, 8, 8], "dims": [0, 2, 3, 1]},
            {"shape": [1, 14, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 1445, 3, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 15, 20, 512], "dims": [0, 3, 1, 2]},
            {"shape": [1, 16, 1, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16, 1370, 80], "dims": [2, 0, 1, 3]},
            {"shape": [1, 16, 16, 1280], "dims": [0, 3, 1, 2]},
            {"shape": [1, 16, 16, 256], "dims": [0, 3, 1, 2]},
            {"shape": [1, 16, 197, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16, 197, 64], "dims": [2, 0, 1, 3]},
            {"shape": [1, 16, 256, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16, 32, 96], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16, 5, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16, 50, 64], "dims": [2, 0, 1, 3]},
            {"shape": [1, 160, 256], "dims": [0, 2, 1]},
            {"shape": [1, 16384, 1, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 16384, 256], "dims": [0, 2, 1]},
            {"shape": [1, 16384, 32], "dims": [0, 2, 1]},
            {"shape": [1, 19200, 1, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 19200, 64], "dims": [0, 2, 1]},
            {"shape": [1, 197, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 197, 16, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 2, 2, 7, 7, 384], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 2, 7, 7, 512], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 2, 8, 8, 384], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 2, 8, 8, 512], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 4096, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 2, 4800, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 2, 7, 2, 7, 384], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 7, 2, 7, 512], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 8, 2, 8, 384], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 2, 8, 2, 8, 512], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 201, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 2048, 8, 160], "dims": [0, 2, 1, 3]},
            {"shape": [1, 2048, 8, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 23, 40, 256], "dims": [0, 3, 1, 2]},
            {"shape": [1, 25, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 1, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 16, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 2, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 256], "dims": [0, 2, 1]},
            {"shape": [1, 256, 5, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 8, 160], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 8, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 8, 96], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 920], "dims": [2, 0, 1]},
            {"shape": [1, 3, 1445, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 3, 16, 16, 16, 16], "dims": [0, 2, 4, 3, 5, 1]},
            {"shape": [1, 30, 40, 320], "dims": [0, 3, 1, 2]},
            {"shape": [1, 300, 1, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 300, 2, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 300, 5, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 300, 8, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 32, 16, 96], "dims": [0, 2, 3, 1]},
            {"shape": [1, 32, 256], "dims": [0, 2, 1]},
            {"shape": [1, 32, 32, 160], "dims": [0, 3, 1, 2]},
            {"shape": [1, 32, 32, 640], "dims": [0, 3, 1, 2]},
            {"shape": [1, 320, 300], "dims": [0, 2, 1]},
            {"shape": [1, 320, 64, 64], "dims": [0, 2, 3, 1]},
            {"shape": [1, 4, 4, 1, 1], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4, 4, 3, 3], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4, 4, 38, 38], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4, 4, 7, 7, 192], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 4, 7, 7, 256], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 4, 8, 8, 192], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 4, 8, 8, 256], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 7, 4, 7, 192], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 7, 4, 7, 256], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 8, 4, 8, 192], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 8, 4, 8, 256], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 4, 91, 1, 1], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4, 91, 3, 3], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4, 91, 38, 38], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 4096, 2, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 4096, 256], "dims": [0, 2, 1]},
            {"shape": [1, 4096, 64], "dims": [0, 2, 1]},
            {"shape": [1, 45, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 4800, 128], "dims": [0, 2, 1]},
            {"shape": [1, 4800, 2, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 49, 3, 24, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [1, 49, 3, 32, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [1, 5, 1024, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 5, 1200, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 5, 16, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 512], "dims": [0, 1]},
            {"shape": [1, 6, 4, 1, 1], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 10, 10], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 19, 19], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 2, 2], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 20, 20], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 3, 3], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 4, 5, 5], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 1, 1], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 10, 10], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 19, 19], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 2, 2], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 20, 20], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 3, 3], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 6, 91, 5, 5], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 60, 80, 128], "dims": [0, 3, 1, 2]},
            {"shape": [1, 64, 256], "dims": [0, 2, 1]},
            {"shape": [1, 64, 3, 24, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [1, 64, 3, 32, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [1, 64, 300], "dims": [0, 2, 1]},
            {"shape": [1, 64, 64, 320], "dims": [0, 3, 1, 2]},
            {"shape": [1, 64, 64, 64], "dims": [0, 3, 1, 2]},
            {"shape": [1, 640, 32, 32], "dims": [0, 2, 3, 1]},
            {"shape": [1, 7, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 7, 7, 1024], "dims": [0, 3, 1, 2]},
            {"shape": [1, 7, 7, 768], "dims": [0, 3, 1, 2]},
            {"shape": [1, 71, 7, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 768, 1500], "dims": [0, 2, 1]},
            {"shape": [1, 768, 196], "dims": [0, 2, 1]},
            {"shape": [1, 768, 49], "dims": [0, 2, 1]},
            {"shape": [1, 768, 8], "dims": [0, 2, 1]},
            {"shape": [1, 8, 2048, 96], "dims": [0, 2, 1, 3]},
            {"shape": [1, 8, 256, 160], "dims": [0, 2, 1, 3]},
            {"shape": [1, 8, 256, 32], "dims": [0, 2, 1, 3]},
            {"shape": [1, 8, 300, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 8, 7, 8, 7, 128], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 8, 7, 8, 7, 96], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 8, 768], "dims": [0, 2, 1]},
            {"shape": [1, 8, 8, 1024], "dims": [0, 3, 1, 2]},
            {"shape": [1, 8, 8, 1280], "dims": [0, 3, 1, 2]},
            {"shape": [1, 8, 8, 7, 7, 128], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 8, 8, 7, 7, 96], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 8, 8, 768], "dims": [0, 3, 1, 2]},
            {"shape": [1, 8, 8, 8, 8, 128], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 8, 8, 8, 8, 96], "dims": [0, 1, 3, 2, 4, 5]},
            {"shape": [1, 9, 12, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 9, 16, 128], "dims": [0, 2, 1, 3]},
            {"shape": [1, 9, 16, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 9, 4, 100, 136], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 4, 13, 17], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 4, 25, 34], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 4, 50, 68], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 4, 7, 9], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 64, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 9, 91, 100, 136], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 91, 13, 17], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 91, 25, 34], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 91, 50, 68], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 9, 91, 7, 9], "dims": [0, 3, 4, 1, 2]},
            {"shape": [1, 96, 56, 56], "dims": [0, 2, 3, 1]},
            {"shape": [1, 96, 64, 64], "dims": [0, 2, 3, 1]},
            {"shape": [10, 10, 12], "dims": [2, 0, 1]},
            {"shape": [10, 10, 16], "dims": [2, 0, 1]},
            {"shape": [10, 10, 8], "dims": [2, 0, 1]},
            {"shape": [15, 15, 6], "dims": [2, 0, 1]},
            {"shape": [16, 49, 3, 6, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [16, 49, 3, 8, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [16, 64, 3, 6, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [16, 64, 3, 8, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [17, 17, 6], "dims": [2, 0, 1]},
            {"shape": [18176, 4544], "dims": [1, 0]},
            {"shape": [197, 197, 12], "dims": [2, 0, 1]},
            {"shape": [197, 197, 16], "dims": [2, 0, 1]},
            {"shape": [2, 2, 12], "dims": [2, 0, 1]},
            {"shape": [2, 2, 16], "dims": [2, 0, 1]},
            {"shape": [2, 2, 6], "dims": [2, 0, 1]},
            {"shape": [2, 2, 8], "dims": [2, 0, 1]},
            {"shape": [2, 7, 2, 7], "dims": [0, 2, 1, 3]},
            {"shape": [2, 8, 2, 8], "dims": [0, 2, 1, 3]},
            {"shape": [4, 49, 3, 12, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [4, 49, 3, 16, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [4, 64, 3, 12, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [4, 64, 3, 16, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [4, 7, 4, 7], "dims": [0, 2, 1, 3]},
            {"shape": [4, 8, 4, 8], "dims": [0, 2, 1, 3]},
            {"shape": [4544, 18176], "dims": [1, 0]},
            {"shape": [4544, 4544], "dims": [1, 0]},
            {"shape": [4672, 4544], "dims": [1, 0]},
            {"shape": [49, 49, 12], "dims": [2, 0, 1]},
            {"shape": [49, 49, 16], "dims": [2, 0, 1]},
            {"shape": [49, 49, 24], "dims": [2, 0, 1]},
            {"shape": [49, 49, 32], "dims": [2, 0, 1]},
            {"shape": [49, 49, 3], "dims": [2, 0, 1]},
            {"shape": [49, 49, 4], "dims": [2, 0, 1]},
            {"shape": [49, 49, 6], "dims": [2, 0, 1]},
            {"shape": [49, 49, 8], "dims": [2, 0, 1]},
            {"shape": [64, 49, 3, 3, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [64, 49, 3, 4, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [64, 64, 12], "dims": [2, 0, 1]},
            {"shape": [64, 64, 16], "dims": [2, 0, 1]},
            {"shape": [64, 64, 24], "dims": [2, 0, 1]},
            {"shape": [64, 64, 3, 3, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [64, 64, 3, 4, 32], "dims": [2, 0, 3, 1, 4]},
            {"shape": [64, 64, 32], "dims": [2, 0, 1]},
            {"shape": [64, 64, 3], "dims": [2, 0, 1]},
            {"shape": [64, 64, 4], "dims": [2, 0, 1]},
            {"shape": [64, 64, 6], "dims": [2, 0, 1]},
            {"shape": [64, 64, 8], "dims": [2, 0, 1]},
            {"shape": [8, 7, 8, 7], "dims": [0, 2, 1, 3]},
            {"shape": [8, 8, 8, 8], "dims": [0, 2, 1, 3]},
            # {"shape": ["s0 + 1", "s0 + 1", 12], "dims": [2, 0, 1]},
            # {"shape": ["s0 + 1", "s0 + 1", 16], "dims": [2, 0, 1]},
            # {"shape": ["s0 + 1", "s0 + 1", 6], "dims": [2, 0, 1]},
            # {"shape": ["s0 + 1", "s0 + 1", 8], "dims": [2, 0, 1]}
            {"shape": [1, 16, 256, 64], "dims": [0, 2, 1, 3]},
            {"shape": [1, 256, 16, 64], "dims": [0, 2, 1, 3]},
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if len(test_vector["permute_specs"]["dims"]) < 2:
            return True, "bfloat8_b not supported with dims  < 2"

    return False, None


def run(
    permute_specs,
    dtype,
    layout,
    *,
    device,
):
    dims = tuple(permute_specs["dims"])
    shape = permute_specs["shape"]

    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    torch_output_tensor = torch.permute(tensor, dims)
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    start_time = start_measuring_time()
    ttnn_output = ttnn.permute(ttnn_tensor, dims)
    e2e_perf = stop_measuring_time(start_time)

    ttnn_output_tensor = ttnn.to_torch(ttnn_output)
    return [check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999), e2e_perf]
