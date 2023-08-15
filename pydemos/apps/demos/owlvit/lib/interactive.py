# Copyright 2023 The pydemos Authors.
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

"""Functions for the interactive generation of Bokeh layouts for OWL-ViT demos.

Forked from the scenic project in:
https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit.
"""

import base64
import functools
import json
from typing import Mapping, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydemos.apps.demos.owlvit.lib import inference
from scenic.model_lib.base_models import box_utils

TEXT_INPUT_PY_CALLBACK_NAME = 'text_input_py_callback'
IMAGE_CONDITIONING_PY_CALLBACK_NAME = 'text_input_py_callback'

IMAGE_COND_NMS_IOU_THRESHOLD = 0.7
IMAGE_COND_MIN_CONF = 0.5


def text_input_py_callback(comma_separated_queries: str, model: inference.Model,
                           image: np.ndarray, threshold: float) -> str:
  """Compute bboxes from queries and return box color updates and legends.

  Args:
    comma_separated_queries: Comma-separated queries. Will be preprocessed.
    model: inference.Model instance
    image: Single uint8 Numpy image of any size. Will be converted to float and
      resized before passing it to the model.
    threshold: Python <class 'float'> value, between 0 and 1.

  Returns:
    JSON compressed dictionary with box color updates and legends.
  """
  queries = [q.strip() for q in comma_separated_queries.split(',')]
  queries = tuple(q for q in queries if q)
  num_queries = len(queries)

  if num_queries > 8:
    # Caps query number to have a certain number of differentiable colors
    # without repeating.
    queries = queries[:8]
    num_queries = len(queries)

  if not num_queries:
    return json.dumps({'color_updates': [], 'legend_text_b64': ''})

  # Compute box display alphas based on prediction scores:
  query_embeddings = model.embed_text_queries(queries)
  top_query_ind, scores = model.get_scores(image, query_embeddings, num_queries)
  alphas = np.zeros_like(scores)
  for i in range(num_queries):
    # Select scores for boxes matching the current query:
    query_mask = top_query_ind == i
    if not np.any(query_mask):
      continue
    query_scores = scores[query_mask]

    # Box alpha is scaled such that the best box for a query has alpha 1.0 and
    # the worst box for which this query is still the top query has alpha 0.1.
    # All other boxes will either belong to a different query, or will not be
    # shown.
    max_score = np.max(query_scores) + 1e-6
    query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
    query_alphas = np.clip(query_alphas, 0.0, 1.0)

    query_alphas[query_alphas < threshold] = 0
    alphas[query_mask] = query_alphas

  # Construct color_updates structure for Bokeh:
  color_updates = []
  for i, (query_ind, alpha) in enumerate(zip(top_query_ind, alphas)):
    color_updates.append((i, _get_query_color(query_ind, float(alpha))))

  # Construct new legend:
  legend_text = _get_query_legend_html(queries)
  # Base64-encode legend text so we don't have to deal with HTML/JSON escaping:
  legend_text_b64 = base64.b64encode(legend_text.encode('utf8')).decode('utf8')

  return json.dumps({
      'color_updates': color_updates,
      'legend_text_b64': legend_text_b64,
  })


def image_conditioning_py_callback(
    geometry_dict: Mapping[str, Union[float, str]],
    *,
    model: inference.Model,
    query_image: np.ndarray,
    target_image: np.ndarray,
) -> str:
  """Updates image conditioning predictions when box is drawn.

  Gets drawn boxes in geometry dict and returns color updates following the
  image conditioned predictions.

  Args:
    geometry_dict: Map containing coordinates of drawn boxes in [y0, x0, y1, x1]
      format with keys 'y0', 'x0', 'y1', 'x1'.
    model: Warmed up instance of `inference.Model`.
    query_image: Array containing the query image for the model.
    target_image: Array containing the target image for the model.

  Returns:
    JSON compressed dictionary with box color updates for the target image and
    the selected box for the query image.
  """
  # Note: Plotly's y coords are swapped compared to TensorFlow:
  box = (geometry_dict['y0'], geometry_dict['x0'], geometry_dict['y1'],
         geometry_dict['x1'])

  query_embedding, best_box_ind = model.embed_image_query(query_image, box)
  _, _, query_image_boxes = model.embed_image(query_image)

  # TODO(alexserrano): Implement multi-query image-conditioned detection.
  num_queries = 1
  top_query_ind, scores = model.get_scores(
      target_image, query_embedding[None, ...], num_queries=1)

  # Apply non-maximum suppression:
  if IMAGE_COND_NMS_IOU_THRESHOLD < 1.0:
    _, _, target_image_boxes = model.embed_image(target_image)
    target_boxes_yxyx = box_utils.box_cxcywh_to_yxyx(target_image_boxes, np)
    for i in np.argsort(-scores):
      if not scores[i]:
        # This box is already suppressed, continue:
        continue
      ious = box_utils.box_iou(
          target_boxes_yxyx[None, [i], :],
          target_boxes_yxyx[None, :, :],
          np_backbone=np)[0][0, 0]
      ious[i] = -1.0  # Mask self-IoU.
      scores[ious > IMAGE_COND_NMS_IOU_THRESHOLD] = 0.0

  # Compute box display alphas based on prediction scores:
  alphas = np.zeros_like(scores)
  for i in range(num_queries):
    # Select scores for boxes matching the current query:
    query_mask = top_query_ind == i
    query_scores = scores[query_mask]
    if not query_scores.size:
      continue

    # Box alpha is scaled such that the best box for a query has alpha 1.0 and
    # the worst box for which this query is still the top query has alpha 0.1.
    # All other boxes will either belong to a different query, or will not be
    # shown.
    max_score = np.max(query_scores) + 1e-6
    query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
    query_alphas[query_alphas < IMAGE_COND_MIN_CONF] = 0.0
    query_alphas = np.clip(query_alphas, 0.0, 1.0)
    alphas[query_mask] = query_alphas

  # Construct color_updates structure for Bokeh:
  color_updates = []
  for i, (query_ind, alpha) in enumerate(zip(top_query_ind, alphas)):
    color_updates.append((i, _get_query_color(query_ind, float(alpha))))

  cx, cy, w, h = (float(c) for c in query_image_boxes[best_box_ind])
  selected_box = {'x': cx, 'y': cy, 'w': w, 'h': h}

  return json.dumps({
      'color_updates': color_updates,
      'selected_box': selected_box,
  })


@functools.lru_cache(maxsize=None)
def _get_query_color(query_ind, alpha=1.0) -> str:
  color = plt.get_cmap('Set1')(np.linspace(0, 1, 10))[query_ind % 10, :3]
  color -= np.min(color)
  color /= np.max(color)
  return mpl.colors.to_hex((color[0], color[1], color[2], alpha),
                           keep_alpha=alpha < 1.0)


def _get_query_legend_html(queries) -> str:
  html = []
  for i, query in enumerate(queries):
    color = _get_query_color(i)
    html.append(
        f'<span style="color: {color}; font-size: 14pt; font-weight: bold;">'
        f'{query}'
        '</span>')
  return '<span style="font-size: 14pt;"> Queries: </span> </br>' + ', '.join(
      html)
