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

"""Utiliy functions to interact with the model.

Contains functions to load the model and apply it to text or image query.
"""

from typing import Tuple

from . import inference
from . import interactive
from . import plotting
import numpy as np
import plotly
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b16 as config_module
import streamlit as st


@st.experimental_singleton(show_spinner=False)
def load_model() -> inference.Model:
  """Loads checkpoint and calls for model warm-up.

  Returns:
      Warmed-up inference.Model instance.
  """
  config = config_module.get_config()

  module = models.TextZeroShotDetectionModule(
      body_configs=config.model.body,
      normalize=config.model.normalize,
      box_bias=config.model.box_bias)

  variables = module.load_variables(config.init_from.checkpoint_path)

  model = inference.Model(config, module, variables)
  model.warm_up()

  return model


def apply_model_to_text_query(
    text_input: str, image: np.ndarray, model: inference.Model,
    threshold: float) -> Tuple[plotly.graph_objects.Figure, str]:
  """Applies model to text and returns the plotly graphic with drawn bboxes.

  Args:
      text_input: Single string with comma-separated queries. Will be processed
        before passing it to the model.
      image: Single uint8 Numpy image of any size. Will be converted to float
        and resized before passing it to the model.
      model: inference.Model instance.
      threshold: Python <class 'float'> value, between 0 and 1.

  Returns:
      plotly_fig: The Plotly layout of the figure.
      legend_html_text: Query legend html text.
  """
  _, _, boxes = model.embed_image(image)

  json_box_color_update = interactive.text_input_py_callback(
      text_input, model, image, threshold)

  plotly_fig, legend_html_text = plotting.generate_plotly_layout(
      image=model.preprocess_image(image),
      boxes=boxes,
      json_box_color_updates=json_box_color_update)

  return plotly_fig, legend_html_text


def apply_model_to_image_query(
    query_image: np.ndarray, target_image: np.ndarray, query_box: np.ndarray,
    model: inference.Model) -> plotly.graph_objects.Figure:
  """Applies model to image and returns the target image with drawn bboxes.

  TODO(alexserrano): implement threshold.
  TODO(alexserrano): implement figure size.
  TODO(alexserrano): implement multiple query boxes (multiquery).

  Args:
    query_image: Array containing the query image for the model.
    target_image: Array containing the target image for the model.
    query_box: Array containing boxes selected from the query image. Will be
      used to query the target image and compute bboxes.
    model: inference.Model instance.

  Returns:
    Plotly target figure with drawn bboxes.
  """

  interactive.IMAGE_COND_MIN_CONF = 0.5
  interactive.IMAGE_COND_NMS_IOU_THRESHOLD = 0.7

  if not query_box:
    plotly_target_fig, _ = plotting.create_image_figure(
        model.preprocess_image(target_image))
    return plotly_target_fig

  _, _, boxes = model.embed_image(target_image)

  json_box_color_updates = interactive.image_conditioning_py_callback(
      query_box[-1],
      model=model,
      query_image=query_image,
      target_image=target_image)

  plotly_target_fig = plotting.create_image_conditional_figure(
      target_image=model.preprocess_image(target_image),
      target_boxes=boxes,
      json_box_color_updates=json_box_color_updates)

  return plotly_target_fig


def normalize_boxes(boxes: np.ndarray, model: inference.Model) -> None:
  """Normalizes boxes to the [0, 1] range required by the model.

  We normalize by model.config.dataset_configs.input_size, which is the
  input dimension required by the model.

  Args:
    boxes: Array of boxes. Each box is expected to be a dictionary containing
      the box coordinates in xyxy format (e.g [{'x0': 1, 'x1': 2, 'y0': 4, 'y1':
      12}]).
    model: Warmed up inference.Model instance.
  """
  for box in boxes:
    x0, x1, y0, y1 = box['x0'], box['x1'], box['y0'], box['y1']
    box['x0'] = x0 / model.config.dataset_configs.input_size
    box['x1'] = x1 / model.config.dataset_configs.input_size
    box['y0'] = y0 / model.config.dataset_configs.input_size
    box['y1'] = y1 / model.config.dataset_configs.input_size
