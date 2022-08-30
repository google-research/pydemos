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

"""Declares Streamlit Component to create an image carousel from a list of URLs.

When clicking an image from the carousel, its URL will be returned as the
component value.
"""

import os
from typing import List, Optional

import streamlit.components.v1 as components

# RELEASE constant, set to False during development to use node dev server,
# set to True on production to use component built version.
_RELEASE = False

if not _RELEASE:
  # `declare_component` returns a function that is used to create
  # instances of the component.
  _component_func = components.declare_component(
      "image_carousel",
      # URL of the local dev server
      url="http://localhost:5000")

else:
  # On production, points to to the component's build directory.
  parent_dir = os.path.dirname(os.path.abspath(__file__))
  build_dir = os.path.join(parent_dir, "frontend/public")
  _component_func = components.declare_component(
      "image_carousel", path=build_dir)


# We create a custom wrapper function that will serve as our component's
# public API, instead of exposing the `declare_component` function.
def image_carousel(image_list: List[str],
                   scroller_height: int = 200,
                   key: Optional[str] = None) -> str:
  """Creates a new instance of "image_carousel" component.

  Args:
    image_list: List of URL strings of images to display.
    scroller_height: Integer describing height in px of the image scroller.
      Defaults to 200px.
    key: An optional string that uniquely identifies this component. If this is
      None, and the component's arguments are changed, the component will be
      re-mounted in the Streamlit frontend and lose its current state.

  Returns:
    The URL link of the last clicked image in the scroller. Returned as a
    string.
  """

  component_value = _component_func(
      image_list=image_list,
      scroller_height=scroller_height,
      key=key,
      default="")  # Default, return empty string (empty URL)

  return component_value
