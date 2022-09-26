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

"""Complete OWL-ViT demo.

Contains text and image conditioned detection and a summary of the OWL-ViT
paper.
"""

import annotated_text
import imageio.v2 as imageio
from lib import inference
from lib import model_wrapper_functions
from lib import paper
from lib import plotting
import numpy as np
from PIL import Image
from pydemos.components.streamlit_image_carousel.src import streamlit_image_carousel
from pydemos.components.streamlit_plotly_event_handler.src import streamlit_plotly_event_handler
import streamlit as st

# ----------------------- CONFIG & INTRO ---------------------- #
# Configurations for the web app
st.set_page_config(
    page_title="OWL-ViT Demo",
    layout="wide",
    # Emoji from the Google Noto Emoji library - subject to Apache License 2.0.
    page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/google/313/owl_1f989.png"
)

# WIP Banner
st.warning(
    "This page is a WIP - Errors may arise and UI is not completely polished, so it might break.",
    icon="⚠️")
st.info(
    "Report any errors, give feedback or suggest features [here](https://docs.google.com/document/d/1jlvOguNI_cfDfYHG6LgDBC-OZqqzaxekXPJC3EmknhQ/edit?usp=sharing).",
    icon="📥")

# Title
st.markdown(
    '<h1 align="center">Simple Open-Vocabulary Object Detection with Vision Transformers</h1>',
    unsafe_allow_html=True)

# Abstract section
paper.abstract()

# Keywords section
keyword_colors = {
    "blue": "#b3cefb",
    "red": "#f1b4af",
    "yellow": "#ffe395",
    "green": "#84f3bd"
}
st.subheader("Keywords")
annotated_text.annotated_text(
    (" open-vocabulary detection transformer", "", keyword_colors["blue"]), " ",
    (" vision transformer", "", keyword_colors["red"]), " ",
    (" zero-shot detection", "", keyword_colors["yellow"]), " ",
    (" image-conditioned detection", "", keyword_colors["blue"]), " ",
    (" one-shot object detection", "", keyword_colors["green"]), " ",
    (" contrastive learning", "", keyword_colors["red"]), " ",
    (" image-text models", "", keyword_colors["yellow"]), " ",
    (" foundation models", "", keyword_colors["green"]), " ",
    (" CLIP", "", keyword_colors["blue"]))

# Constant list of image URLs showcased in the carousels
IMAGE_URL_LIST = [
    ("https://burst.shopifycdn.com/photos/healthy-breakfast-time.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Sarah Pflug", "Burst Some Rights Reserved"),
    ("https://cdn.magdeleine.co/wp-content/uploads/2021/05/StockSnap_DXLNXCCDGP.jpg",
     "Photo by Lisa Fotios", "CC0"),
    ("https://burst.shopifycdn.com/photos/musical-and-rustic-domestic-objects-inside-a-caravan.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Angelique Downing", "Burst Some Rights Reserved"),
    ("https://burst.shopifycdn.com/photos/a-picture-reads-live-laugh-love-joined-by-random-objects.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Andréa Felsky Schmitt", "Burst Some Rights Reserved"),
    ("https://burst.shopifycdn.com/photos/startup-desk.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Matthew Henry", "CC0"),
    ("https://burst.shopifycdn.com/photos/office-wooden-desk-workspace.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Matthew Henry", "Burst Some Rights Reserved"),
    ("https://burst.shopifycdn.com/photos/morning-journal.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Matthew Henry", "CC0"),
    ("https://burst.shopifycdn.com/photos/bedroom-with-heart-pillows.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Shopify Partners", "Burst Some Rights Reserved"),
    ("https://burst.shopifycdn.com/photos/comfortable-living-room-cat.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Brodie", "CC0"),
    ("https://burst.shopifycdn.com/photos/yellow-pillow-bedside-table.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Matthew Henry", "CC0"),
    ("https://burst.shopifycdn.com/photos/dinner-party.jpg?width=925&format=pjpg&exif=1&iptc=1",
     "Photo by Matthew Henry", "CC0"),
]


# ----------------------- TEXT CONDITIONED DEMO ---------------------- #
def run_text_conditioned_demo() -> inference.Model:
  """Instantiates and warms up `inference.Model` and runs text conditioned demo.

  Returns:
      Warmed up instance of `inference.Model`.
  """

  st.subheader("Text Conditioned Demo")
  target_tab, upload_tab = st.tabs(
      ["Choose the target image", "Upload the target image"])

  # Instantiates target image to avoid reference before assignment.
  image = None

  with target_tab:

    # Instantiates model.
    with st.spinner("Warming up the model. Doing `jit` compilation"):
      model = model_wrapper_functions.load_model()

    st.info("Select an image to use as target here: ", icon="🎯")
    selection = streamlit_image_carousel.image_carousel(
        image_list=IMAGE_URL_LIST, scroller_height=200, key="text")
    if selection:
      image = np.array(imageio.imread(selection))

  with upload_tab:
    st.warning(
        "While the hosting is still a WIP, there might be errors when uploading your own images.",
        icon="⚠️")
    img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
      image = np.array(Image.open(img_file_buffer))

  # Distributes columns to have a big image.
  left, right = st.columns([1, 3])

  if image is not None:
    with left:
      text_input = st.text_input("Write comma-separated queries here:")
      query_text_container = st.container()
      threshold = st.slider(
          label="Select the threshold for your bounding boxes:",
          min_value=0.,
          max_value=1.,
          value=0.1,
          step=0.01,
          key="threshold_slider")

    with right:
      with st.spinner("Embedding image and queries..."):
        # Get image with bboxes and query legend html text.
        result, legend_html_text = model_wrapper_functions.apply_model_to_text_query(
            text_input, image, model, threshold)
      st.plotly_chart(result, config={"staticPlot": True})

    with query_text_container:
      st.markdown(legend_html_text, unsafe_allow_html=True)

  return model


# ----------------------- IMAGE CONDITIONED DEMO ---------------------- #
def run_image_conditioned_demo(model: inference.Model):
  """Runs image conditioned demo.

  Args:
    model: Warmed up instance of inference model.
  """

  st.subheader("Image Conditioned Demo")

  query_tab, query_upload_tab = st.tabs(
      ["Choose the query image", "Upload the query image"])

  # Instantiates query and target images to avoid reference before assignment.
  query_image = target_image = None

  with query_tab:
    st.info("Select an image to use as query here:", icon="✏️")
    query_image_selection = streamlit_image_carousel.image_carousel(
        image_list=IMAGE_URL_LIST, scroller_height=200, key="query_image")

    st.info("Select an image to use as target here:", icon="🎯")
    target_image_selection = streamlit_image_carousel.image_carousel(
        image_list=IMAGE_URL_LIST, scroller_height=200, key="target_image")

    if query_image_selection and target_image_selection:
      query_image = np.array(imageio.imread(query_image_selection))
      target_image = np.array(imageio.imread(target_image_selection))

  with query_upload_tab:
    st.warning(
        "While the hosting is still a WIP, there might be errors when uploading your own images.",
        icon="⚠️")

    img_cond_buffer_query = st.file_uploader(
        "Query image", type=["png", "jpg", "jpeg"])
    img_cond_buffer_target = st.file_uploader(
        "Target image", type=["png", "jpg", "jpeg"])

    if img_cond_buffer_query is not None and img_cond_buffer_target is not None:
      query_image = np.array(Image.open(img_cond_buffer_query))
      target_image = np.array(Image.open(img_cond_buffer_target))

  if query_image is not None and target_image is not None:
    plotly_query_fig, _ = plotting.create_image_figure(
        model.preprocess_image(query_image))
    plotly_query_fig.update_layout(
        dragmode="drawrect",
        newshape=dict(line_color="red"),
        # Removes all default buttons in the modebar.
        modebar_remove=[
            "autoscale", "lasso", "pan", "reset", "select", "toImage",
            "toggleHover", "toggleSpikelines", "togglehover",
            "togglespikelines", "toimage", "zoom", "zoomin", "zoomout"
        ],
        # Disables annotations when hovering the image.
        hovermode=False,
    )

    left, right = st.columns(2)

    with left:
      # Attach event handler to the plot and get boxes (event data).
      box = streamlit_plotly_event_handler.relayout_event_handler(
          plot=plotly_query_fig, override_height=600)
      model_wrapper_functions.normalize_boxes(box, model)

    # Applies model to image query.
    plotly_target_fig = model_wrapper_functions.apply_model_to_image_query(
        query_image=query_image,
        target_image=target_image,
        query_box=box,
        model=model)

    with right:
      plotly_target_fig.update_layout(margin=dict(l=30, r=30, b=0, t=10))
      st.plotly_chart(
          plotly_target_fig, config={
              "staticPlot": True,
              "responsive": False
          })


# Sets containers for rendering the demos after rendering the rest of the page.
text_conditioning_container = st.container()
image_conditioning_container = st.container()

# ----------------------- PAPER ---------------------- #
paper.introduction()

paper.related_work()

paper.method()

paper.experiments()

paper.conclusion()

paper.references()

# ----------------------- PAPER ---------------------- #
# Runs both demos after rendering the paper.
# Renders the demos inside the previously declared containers.
with text_conditioning_container:
  model_instance = run_text_conditioned_demo()

with image_conditioning_container:
  run_image_conditioned_demo(model_instance)
