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

"""OWL-ViT plotting functions.

Forked from the scenic project in:
https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit.
"""

import base64
import json
from typing import Optional, Tuple

import matplotlib as mpl
import numpy as np
import plotly
import plotly.express as px
from scenic.model_lib.base_models import box_utils
import streamlit as st


@st.experimental_singleton(show_spinner=False)
def generate_plotly_layout(
    image: np.ndarray, boxes: np.ndarray,
    json_box_color_updates: str) -> Tuple[plotly.graph_objects.Figure, str]:
  """Creates a Plotly figure for interactive text-conditional detection.

  Args:
    image: Image to detect objects in.
    boxes: All predicted boxes for the image, in [cx, cy, w, h] format.
    json_box_color_updates: Color box updates in json format.

  Returns:
    plot: The Plotly figure of the image.
    legend_html_text: Query legend html text.
  """

  plot, padding_compensation_factor = create_image_figure(
      image, return_padding=True)

  # Declares all boxes transparent (using RGBA).
  colors = ['#00000000' for _ in boxes]

  result_json = json.loads(json_box_color_updates)
  color_updates = result_json['color_updates']

  legend_html_text = base64.b64decode(
      result_json['legend_text_b64']).decode('utf-8')

  for i in range(len(color_updates)):
    # Element 0 is the box index, element 1 is the hex color:
    colors[color_updates[i][0]] = color_updates[i][1]

  _plot_boxes(plot, boxes, colors, padding_compensation_factor)

  return plot, legend_html_text


@st.experimental_singleton(show_spinner=False)
def create_image_conditional_figure(
    target_image: np.ndarray, target_boxes: np.ndarray,
    json_box_color_updates: str) -> plotly.graph_objects.Figure:
  """Creates Plotly figures for interactive image-conditioned detection.

  Gets query, target image, boxes and color updates. Creates Plotly figures for
  both images and plots the updated boxes following the color updates.

  Args:
    target_image: Image in which to detect objects.
    target_boxes: Predicted boxes for the target image ([cx, cy, w, h] format).
    json_box_color_updates: Color updates for predicted boxes.

  Returns:
    The Plotly layout of the target image with drawn bboxes.
  """
  plot, padding_factor_target = create_image_figure(
      target_image, return_padding=True)

  # Initializes all box colors transparent (in RGBA format).
  colors = ['#00000000' for _ in target_boxes]

  result_json = json.loads(json_box_color_updates)
  color_updates = result_json['color_updates']

  for i in range(len(color_updates)):
    # Element 0 is the box index, element 1 is the hex color:
    colors[color_updates[i][0]] = color_updates[i][1]

  _plot_boxes(
      plot,
      target_boxes,
      colors,
      padding_compensation_factor=padding_factor_target)

  return plot


def create_image_figure(
    image: np.ndarray,
    title: Optional[str] = '',
    return_padding: Optional[bool] = False
) -> Tuple[plotly.graph_objects.Figure, Optional[int]]:
  """Creates a Plotly figure showing an image.

  TODO(alexserrano): responsivity.
  TODO(alexserrano): option to change figure size in plotly.

  Args:
    image: Image array.
    title: (Optional) Title of the plot.
    return_padding: (Optional) Boolean whether to return padding.

  Returns:
    fig: Plotly figure with the input image.
    padding_compensation_factor: (Optional) Difference between the real
                                  width/height and the padded width/height
  """
  # Determine relative width and height from padding. We assume that padding is
  # added on the bottom or right and has value 0.5:
  width = np.mean(np.any(image[..., 0] != 0.5, axis=0))
  height = np.mean(np.any(image[..., 0] != 0.5, axis=1))

  real_width = int(width * np.shape(image)[0])
  real_height = int(height * np.shape(image)[1])

  # Creates Plotly figure from the image.
  fig = px.imshow(
      image[:real_height, :real_width, ...],
      width=real_width,
      height=real_height,
      title=title)
  # Deletes axes.
  fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
  # And forces the margins to stay constant.
  fig.update_layout(margin=dict(l=30, r=30, b=0, t=0))

  if return_padding:
    padding_compensation_factor = (abs(np.shape(image)[0] - real_height),
                                   abs(np.shape(image)[1] - real_width))
    return fig, padding_compensation_factor

  return fig, None


def _plot_boxes(fig: plotly.graph_objects.Figure,
                boxes: np.ndarray,
                colors: np.ndarray,
                padding_compensation_factor: int = (0, 0),
                line_width: int = 3) -> None:
  """Adds boxes to the provided Plotly figure.

  Args:
    fig: Plotly figure where to print the boxes.
    boxes: Array of boxes in [y0, x0, y1, x1] format.
    colors: Array of colors in RGBA format.
    padding_compensation_factor: Tuple of height/width padding compensation.
    line_width: Integer that will determine line width for the bboxes drawn.
  """
  boxes = box_utils.box_cxcywh_to_yxyx(boxes, np)
  for box, color in zip(boxes, colors):
    y0, x0, y1, x1 = box

    # Box from [0,1] to [[0, width/height] compensating for padding suppression.
    (fig_height_diff, fig_width_diff) = padding_compensation_factor
    y0, y1 = y0 * (fig.layout.height +
                   fig_height_diff), y1 * (fig.layout.height + fig_height_diff),
    x0, x1 = x0 * (fig.layout.width + fig_width_diff), x1 * (
        fig.layout.width + fig_width_diff)

    # Box assigned color is rgba, but plotly only
    # receives rgb and opacity is treated apart.
    r, g, b, a = mpl.colors.to_rgba(color)

    # To save time, do not draw if opacity is very close to zero.
    if a >= 1e-10:
      fig.add_shape(
          type='rect',
          x0=x0,
          x1=x1,
          y0=y0,
          y1=y1,
          xref='x',
          yref='y',
          line=dict(color=mpl.colors.to_hex((r, g, b)),),
          line_width=line_width,
          opacity=a)
