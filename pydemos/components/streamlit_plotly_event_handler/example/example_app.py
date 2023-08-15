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

"""Example app to showcase streamlit_plotly_event_handler Streamlit Component.

Generates a Plotly plot from an image URL and listens to relayout events that
get returned from the component and printed on the screen.
"""

import imageio.v2 as imageio
import numpy as np
import plotly.express as px
import streamlit as st
import streamlit_plotly_event_handler as speh


# Get image as array
example_image_url = "https://burst.shopifycdn.com/photos/dinner-party.jpg?width=925&format=pjpg&exif=1&iptc=1"
image = np.array(imageio.imread(example_image_url))

# Transform into a Plotly plot.
fig = px.imshow(image)

# Delete axes from the plot.
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

# No margins and allow annotation drawing for relayout event.
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode="drawrect",
    newshape=dict(line_color="red"))

# Listen to relayout event and print boxes returned.
boxes = speh.relayout_event_handler(fig)
st.write(boxes)
