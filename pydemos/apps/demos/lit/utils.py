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

"""Utils."""

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from vit_jax import models


@st.cache()
def get_models():
  return [name for name in models.model_configs.MODEL_CONFIGS
          if name.startswith('LiT')]


@st.experimental_singleton()
def load_model(model_name):
  return models.get_model(model_name)


@st.experimental_singleton(show_spinner=False)
def load_variables(model_name):
  lit_model = load_model(model_name)
  lit_variables = lit_model.load_variables()  # Downloads from cloud
  # lit_variables = lit_model.load_variables(path=model_name +'.npz')
  image_preprocessing = lit_model.get_image_preprocessing()
  return lit_model, lit_variables, image_preprocessing


# Tokenizer couldn't be cached.
def load_tokenizer(model):
  return model.get_tokenizer()


@st.cache(show_spinner=False)
def embed_images(model_name, lit_variables, input_image, image_preprocessing):
  model = load_model(model_name)
  images = image_preprocessing(np.array(input_image))
  zimg, _, out = model.apply(lit_variables, images=images)
  return zimg, out


# Can you compile this in jax? Or does it already compile automatically?
def run_inference(zimg, ztxt, out):
  return np.array(jax.nn.softmax(out['t'] * ztxt @ zimg.T, axis=0))


# --------- ZERO-SHOT ----------
@st.experimental_singleton
def load_embedding(model_name, dataset_name):
  """Load precomputed text embedding of selected datasets."""
  file = open(f'datasets-labels/processed/{model_name}_datasets_ztxt.npy',
              'rb')
  alldatasets = np.load(file, allow_pickle=True)
  ztxt = alldatasets.item().get(dataset_name)
  file.close()
  dataset = open('datasets-labels/raw/' + dataset_name + '-classes.txt',
                 'rb').read().decode('utf-8')
  classnames = process_dataset(dataset)
  return ztxt, classnames


@st.experimental_singleton
def process_dataset(classes):
  classes = classes.splitlines()
  for i in range(len(classes)):
    classes[i] = classes[i].split(',')
    classes[i] = classes[i][0]  # Temporary to keep the first name of the class
  return classes


def process_embedding(model_name, dataset):
  """Process text embedding of uploaded datasets."""
  model, lit_variables, _ = load_variables(model_name)
  tokenizer = load_tokenizer(model)
  prompts = [
      'itap of a {}.',
      'a photo of the {}.',
      'art of the {}.',
      '{}',
  ]
  classnames = process_dataset(dataset)
  if len(classnames) > 1000:
    return 'Your dataset is too large. Maximum is 1k labels', 0
  texts = []
  for classname in classnames:
    for prompt in prompts:
      texts.append(prompt.format(classname))
  tokens = tokenizer(texts)
  _, ztxt, _ = model.apply(lit_variables, tokens=tokens)
  ztxt = jnp.reshape(ztxt, (len(classnames), len(prompts), -1))
  ztxt = jnp.mean(ztxt, axis=1)
  return ztxt, classnames


def get_top5(probs, classnames):
  """Returns plotly figure showing the 5 labels with highest softmax score."""
  clean_probs = []
  for i in range(len(probs)):
    clean_probs.append(probs[i][0])  # Workaround to delete intermediate lists
  index_top5 = np.argsort(clean_probs, -5)[-5:]
  x = []
  y = []
  for index in index_top5:
    x.append(float(round(clean_probs[index] * 100, 2)))
    y.append(classnames[index])

  fig = go.Figure(data=[go.Bar(x=x, y=y, orientation='h', text=x)])
  fig.update_traces(
      marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
      marker_line_width=1.5, opacity=0.6)
  return fig


# ---------- PAPER --------------
def write_intro():
  st.markdown('''<p align="justify">This page is an interactive demo of the
Google AI blog post <a href="https://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html">
LiT: adding language understanding to image models</a>. Please refer
to that page for a detailed explanation how a LiT model works.
<br>
<br>
Below you can choose to use an example image or upload your own and then write
free-form text prompts that are matched to the image. Once you press the
"compute" button, a text encoder will compute embeddings for the provided text,
and the similarity of these text embeddings to the image embedding will be displayed.
<br>
<br>
The prompts can be used to classify an image into multiple categories, listing
each category individually with a prompt "an image of a X". But you can also
probe the model interactively with more detailed prompts, comparing the different
results when small details change in the text.
<br>
<br>
Please use this demo responsibly. The models will always compare the image to
the prompts you provide, and it is therefore trivial to construct situations
where the model picks from a bunch of bad options.</p>'''
              , unsafe_allow_html=True)
