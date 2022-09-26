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

"""Example app to showcase streamlit_image_carousel Streamlit Component.

Generates an image carousel from a list of URLs using the Svelte framework.
Clicking on an image will return its URL as the component value.
"""

# Demo app for development purposes.
import streamlit as st
from streamlit_image_carousel import image_carousel

st.subheader("Image carousel component")

image_url_list = [
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

selectedImageUrl = image_carousel(
    image_list=image_url_list, scroller_height=200)

if selectedImageUrl:
  st.image(selectedImageUrl)
