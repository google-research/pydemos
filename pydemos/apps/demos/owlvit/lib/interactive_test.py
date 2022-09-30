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

"""Tests for interactive."""

from unittest import mock

from absl.testing import absltest
import numpy as np
from pydemos.apps.demos.owlvit.lib import interactive


def _mock_tokenize(text, max_len):
  del text
  return np.zeros((max_len,), dtype=np.int32)


class InteractiveTest(absltest.TestCase):

  def test_text_input_py_callback(self):
    model = mock.Mock()
    model.embed_text_queries = mock.Mock(
        return_value=np.zeros((50, 512), dtype=np.int32))
    model.get_scores = mock.Mock(
        return_value=(np.array([0, 1]), np.array([0.9, 0.0])))

    comma_separated_queries = 'foo, bar, baz'

    out = interactive.text_input_py_callback(
        comma_separated_queries=comma_separated_queries,
        model=model,
        image=np.zeros((128, 64, 3), dtype=np.uint8),
        threshold=0)

    expected_callback_out = (
        '{"color_updates": [[0, "#ff0003ff"], [1, "#008cff00"]], '
        '"legend_text_b64": '
        '"PHNwYW4gc3R5bGU9ImZvbnQtc2l6ZTogMTRwdDsiPiBRdWVyaWVzOiA8L3NwYW4+IDwvYnI+PHNwYW4gc3R5bGU9ImNvbG9yOiAjZmYwMDAzOyBmb250LXNpemU6IDE0cHQ7IGZvbnQtd2VpZ2h0OiBib2xkOyI+Zm9vPC9zcGFuPiwgPHNwYW4gc3R5bGU9ImNvbG9yOiAjMDA4Y2ZmOyBmb250LXNpemU6IDE0cHQ7IGZvbnQtd2VpZ2h0OiBib2xkOyI+YmFyPC9zcGFuPiwgPHNwYW4gc3R5bGU9ImNvbG9yOiAjMDhmZjAwOyBmb250LXNpemU6IDE0cHQ7IGZvbnQtd2VpZ2h0OiBib2xkOyI+YmF6PC9zcGFuPg=="}'
    )
    self.assertEqual(expected_callback_out, out)

  def test_image_conditioning_py_callback(self):
    model = mock.Mock()
    model.embed_image = mock.Mock(return_value=(None, None, np.zeros((2, 4))))
    model.embed_image_query = mock.Mock(
        return_value=(np.zeros(512), np.zeros((), dtype=np.int32)))
    model.get_scores = mock.Mock(
        return_value=(np.array([0, 1]), np.array([0.9, 0.0])))

    out = interactive.image_conditioning_py_callback(
        {
            'x0': 0.1,
            'x1': 0.2,
            'y0': 0.3,
            'y1': 0.4
        },
        model=model,
        query_image=np.zeros((128, 64, 3), dtype=np.uint8),
        target_image=np.zeros((128, 128, 3), dtype=np.uint8))

    expected_out = ('{"color_updates": [[0, "#ff0003ff"], [1, "#008cff00"]], '
                    '"selected_box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}}')

    self.assertEqual(out, expected_out)


if __name__ == '__main__':
  absltest.main()
