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

"""Streamlit app to showcase examples of streamlit built-in components."""

import io

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from pydemos.components.streamlit_image_carousel.src import streamlit_image_carousel
import requests
import streamlit as st
import utils

# ----------------------- CONFIG & INTRO ---------------------- #
# Configurations for the web app
st.set_page_config(
    page_title='LiT: Zero-Shot Transfer with Locked-image text Tuning',
    layout='wide',
    page_icon=':fire:')

# Title
st.title(':fire: LiT: Zero-Shot Transfer with Locked-image text Tuning')

# Share buttons
st.markdown('''
[![Twitter][share_badge]][share_link] [![GitHub][github_badge]][github_link]

[share_badge]: https://badgen.net/badge/icon/twitter?icon=twitter&color=black&label
[share_link]: https://twitter.com/googleai

[github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
[github_link]: https://github.com/google-research/vision_transformer
''')

# Intro section
utils.write_intro()

# Model Selection and processing
model_name = st.radio(
    'Choose a model. The size of the model will affect loading times.',
    ['LiT-B16B', 'LiT-L16S', 'LiT-L16Ti', 'LiT-L16L'], horizontal=True)
with st.spinner(f'Warming up {model_name}...'):
  model, lit_variables, image_preprocessing = utils.load_variables(model_name)
  tokenizer = utils.load_tokenizer(model)

# List of image URLs showcased in the carousels
image_url_list = [
    ('https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg',
     'Photo by OpenAI', 'CC0'),
    ('https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg',
     'Photo by OpenAI', 'CC0'),
    ('https://cdn.pixabay.com/photo/2022/03/25/23/15/shetland-ponies-7091959_1280.jpg',
     'No attribution required', 'Pixabay Some Rights Reserved'),
    ('https://cdn.magdeleine.co/wp-content/uploads/2021/05/StockSnap_DXLNXCCDGP.jpg',
     'Photo by Lisa Fotios', 'CC0'),
    ('https://burst.shopifycdn.com/photos/a-regal-white-dog-in-a-flower-bonnet.jpg?width=1850&format=pjpg&exif=1&iptc=1',
     'Photo by Samantha Hurley', 'Burst Some Rights Reserved'),
    ('https://cdn.pixabay.com/photo/2021/09/27/06/10/animal-6659568_1280.jpg',
     'No attribution required', 'Pixabay')
    ]


# Dictionary with urls and text prompts
url_to_texts = {
    'https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg':
        ['An astronaut riding a horse in space', 'a space knight', 'a pegasus',
         'a man riding a white seal', 'a man disguised as a horse'],
    'https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg':
        ['an apple', 'an ipod', 'granny smith',
         'an apple with a note saying "ipod"', 'an adversarial attack'],
    'https://cdn.pixabay.com/photo/2022/03/25/23/15/shetland-ponies-7091959_1280.jpg':
        ['a shetland pony mare and her foal', 'a horse and her baby',
         'a horse', 'a baby horse', 'grass'],
    'https://cdn.magdeleine.co/wp-content/uploads/2021/05/StockSnap_DXLNXCCDGP.jpg':
        ['a plant', 'a pot', 'four cactus',
         'three pots', 'four pots and 4 plants'],
    'https://burst.shopifycdn.com/photos/person-holds-a-book-over-a-stack-and-turns-the-page.jpg?width=1850&format=pjpg&exif=1&iptc=1':
    ['hands holding a book', 'a book', 'hands', 'reading', 'reading a book'],
    'https://burst.shopifycdn.com/photos/a-regal-white-dog-in-a-flower-bonnet.jpg?width=1850&format=pjpg&exif=1&iptc=1':
    ['a white dog with a flowery bonnet', 'a cat with a flowery bonnet',
     'a dog', 'will you marry me?', 'a cat'],
    'https://cdn.pixabay.com/photo/2021/09/27/06/10/animal-6659568_1280.jpg':
    ['a cat behind a ladder', 'a cat', 'a dog', 'a ladder',
     'a cat under a ladder']
}

example_image_url = streamlit_image_carousel.image_carousel(
    image_list=image_url_list, scroller_height=200)

example_image = io.BytesIO(requests.get(example_image_url).content)
input_image = [np.array(Image.open(example_image))]
upload_image = st.file_uploader('Upload your own image', type=['jpg', 'jpeg'])
if upload_image is not None:
  input_image = [np.array(Image.open(upload_image))]
with st.spinner('Embedding the image.'):
  zimg, out = utils.embed_images(model_name, lit_variables, input_image,
                                 image_preprocessing)


def run_lit_demo():
  """Runs the original lit demo with user prompt inputs."""
  left, mid = st.columns([2, 3])
  left.image(input_image[0])
  if upload_image is None:
    texts = url_to_texts[example_image_url]
  else:
    texts = ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5']
  for i in range(len(texts)):
    texts[i] = mid.text_input('', value=texts[i])
  compute = left.button('Compute 🔥', key='compute')
  if compute:
    with st.spinner(''):
      tokens = tokenizer(texts)
      _, ztxt, _ = model.apply(lit_variables, tokens=tokens)
      probs = utils.run_inference(zimg, ztxt, out)
    st.subheader('Results:')
    x = []
    y = []
    for s in range(len(probs[0])):
      for i in range(len(texts)):
        x.insert(0, float(round(probs[i][s]*100, 2)))
        y.insert(0, texts[i])
    fig = go.Figure(data=[go.Bar(x=x, y=y, orientation='h', text=x)])
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)


def run_zero_shot_demo():
  """Runs the zero shot demo on a set of labels from a chosen dataset."""
  left, right = st.columns([2, 5])
  left.image(input_image[0])
  dataset_name = right.selectbox(
      'Select a list of class names from a dataset to run Zero-Shot on them.',
      ('upload labels', 'cifar10', 'stl10', 'cifar100', 'flowers102',
       'food101', 'ucf101', 'imagenet', 'kinetics700',
       'write labels'))  # Write labels TBD and discussed (maybe not necessary)
  right.warning(
      '''This version only supports LiT-B16B for predefined datasets on
Zero-Shot (WIP). You can still upload your own dataset on any model.''',
      icon='⚠️')
  if dataset_name != 'upload labels' and dataset_name != 'write labels':
    ztxt, classnames = utils.load_embedding(model_name, dataset_name)
    probs = utils.run_inference(zimg, ztxt, out)
    fig = utils.get_top5(probs, classnames)
    right.plotly_chart(fig, use_container_width=True)
  else:
    upload_dataset = right.file_uploader(
        'Upload your own dataset with each class label on a different line.',
        type=['txt'])
    if upload_dataset:
      upload_dataset = upload_dataset.getvalue().decode('utf-8')
      ztxt, classnames = utils.process_embedding(model_name, upload_dataset)
      if isinstance(ztxt, str):
        right.write(ztxt)
      else:
        probs = utils.run_inference(zimg, ztxt, out)
        fig = utils.get_top5(probs, classnames)
        right.plotly_chart(fig, use_container_width=True)


# Future work: Fewshot tab
# def run_few_shot_demo():
#  st.write('Few Shot TBD')

prompt_tab, zeroshot_tab = st.tabs(
    ['Prompts', 'Zero-Shot'])

with prompt_tab:
  run_lit_demo()
with zeroshot_tab:
  run_zero_shot_demo()
