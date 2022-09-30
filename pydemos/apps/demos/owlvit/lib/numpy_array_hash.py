# Copyright 2022 The pydemos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a function to deterministically hash numpy arrays."""

import hashlib
import numpy as np


def hash_array(arr: np.ndarray):
  """Returns a deterministic hash for the numpy array `arr`.

  Args:
    arr: Numpy array that will be hashed.

  Returns:
    Hashed numpy array.
  """

  array_hash = hashlib.black2b(arr.tobytes(), digest_size=20)

  # Adds array shape as part of the hash to be able to differentiate
  # transposed arrays and other transformations.
  for dim in arr.shape:
    array_hash.update(dim.tobytes(4, byteorder='big'))

  return array_hash.digest()
