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

"""Declares Streamlit Component to listen to relayout events in Plotly plots."""

import json
import os
from typing import Optional

import plotly
import streamlit.components.v1 as components

# RELEASE constant, set to False during development to use node dev server,
# set to True on production to use component built version.
_RELEASE = False

if not _RELEASE:
  # `declare_component` returns a function that is used to create
  # instances of the component.
  _component_func = components.declare_component(
      "plotly_relayout_event_handler",
      # URL of the local dev server
      url="http://localhost:3001",
  )
else:
  # On production, points to to the component's build directory:
  parent_dir = os.path.dirname(os.path.abspath(__file__))
  build_dir = os.path.join(parent_dir, "frontend/build")
  _component_func = components.declare_component(
      "plotly_relayout_event_handler", path=build_dir)


# We create a custom wrapper function that will serve as our component's
# public API, instead of exposing the `declare_component` function.
def relayout_event_handler(
    plot: plotly.graph_objects.Figure,
    override_height: Optional[str | int] = 450,
    override_width: Optional[str | int] = "100%",
    key: Optional[str] = None,
):
  """Creates a new instance of "plotly_relayout_event_handler" component.

  Args:
    plot: The Plotly figure that will be rendered and listened for relayout
      events.
    override_height: String or integer to override component height. If it is an
      integer, default unit is pixel. String is used to express height in %.
      Defaults to 450px.
    override_width: String or integer to override width. If it is an integer,
      default unit is pixel. String is used to express height in %. Defaults to
      100% (whole width of iframe).
    key: An optional string that uniquely identifies this component. If this is
      None, and the component's arguments are changed, the component will be
      re-mounted in the Streamlit frontend and lose its current state.

  Returns:
    List of dictionaries containing drawn shape coordinates in xxyy
    format.

    Details can be found here (plotly.js docs):
        https://plotly.com/javascript/plotlyjs-events/#event-data
    and here (react-plotly.js docs):
        https://github.com/plotly/react-plotly.js/#event-handler-props

    Format of dict:
        {
            x0: float (value of smallest x coord),
            x1: float (value of largest x coord),
            y0: float (value of smallest y coord),
            y1: float (value of largest y coord)
        }
  """

  # kwargs will be exposed to frontend in "args"
  component_value = _component_func(
      plot_obj=plot.to_json(),
      override_height=override_height,
      override_width=override_width,
      key=key,
      default="[]",  # Default, return empty list in JSON format
  )

  # Parse component_value since it's JSON and return to Streamlit
  return json.loads(component_value)
