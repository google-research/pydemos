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
import requests
import streamlit as st
import utils


# Sets page title and layout
st.set_page_config(
    page_title='LiT: Zero-Shot Transfer with Locked-image text Tuning',
    layout='wide',
    page_icon='https://www.jrieke.com/assets/images/streamlit.png')

st.title('LiT: Zero-Shot Transfer with Locked-image text Tuning')
utils.write_intro()

model_name = st.radio(
    'Choose a model. The size of the model will affect loading times.',
    ['LiT-B16B', 'LiT-L16S', 'LiT-L16Ti', 'LiT-L16L'], horizontal=True)

with st.spinner(f'Warming up {model_name}...'):
  model, lit_variables, image_preprocessing = utils.load_variables(model_name)
  tokenizer = utils.load_tokenizer(model)

example_image = io.BytesIO(
    requests.get(
        'https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg'
        ).content)
input_image = [np.array(Image.open(example_image))]
upload_image = st.file_uploader('Upload your own image', type=['jpg', 'jpeg'])
if upload_image is not None:
  input_image = [np.array(Image.open(upload_image))]
with st.spinner('Embedding the image.'):
  zimg, out = utils.embed_images(model_name, lit_variables, input_image,
                                 image_preprocessing)


def run_lit_demo():
  """Runs the original lit demo with user prompt inputs."""
  left, mid, right = st.columns([2, 4, 1])
  left.image(input_image[0])
  compute = mid.button('Compute 🔥', key='compute')
  texts = [
      'An astronaut riding a horse in space', 'a space knight', 'a pegasus',
      'a man riding a white seal', 'a man disguised as a horse'
  ]
  for i in range(len(texts)):
    texts[i] = mid.text_input('', value=texts[i])
  if compute:
    with right:
      with st.spinner(''):
        tokens = tokenizer(texts)
        _, ztxt, _ = model.apply(lit_variables, tokens=tokens)
        probs = utils.run_inference(zimg, ztxt, out)
    right.write('')
    right.write('')
    for s in range(len(probs[0])):
      for i in range(len(texts)):
        right.write('')
        right.write('')
        right.write('')
        right.markdown(str(round(float(probs[i][s]*100), 3)) + ' %')


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


def run_few_shot_demo():
  st.write('Few Shot TBD')

prompt_tab, zeroshot_tab, fewshot_tab = st.tabs(
    ['Prompts', 'Zero-Shot', 'Few-Shot'])

with prompt_tab:
  run_lit_demo()
with zeroshot_tab:
  run_zero_shot_demo()
with fewshot_tab:
  run_few_shot_demo()
