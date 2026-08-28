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

"""This script produces descriptive statistics, and creates a causal impact
      model with the file provided in File_Upload.


model.py creates a descriptive chart for the file encoded at File_Upload.py.
  It then prompts the user for relevant inputs in order to produce a
  CausalImpact model, and a .png chart that's displayed in the UI. The user can
  interactively tweak inputs with streamlit elements frovided in the GUI.
"""
import datetime as dt
import os
import time

from causalimpact import CausalImpact
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# GUI Elements Start Loading ::
st.title('🧪 Brew your Causal Model')
file_path = 'data_file.csv'


# Step 01 :: load data
def load_file(message=False):
  """Loads file into pandas dataframe.

  Args:
    message: A bool representing logic to produce last modified status of file.

  Returns:
    A streamlit dataframe element of the user's uploaded file, or throws an
    error if invalid.

  Raises:
    FileNotFoundError: An error ocurred accessing the .csv file.
  """
  try:
    data = pd.read_csv(
        file_path, index_col=0, thousands=',', parse_dates=['date']
    )

    if message:
      last_modified = os.stat(file_path).st_mtime
      st.text(f'Last Modified: {time.ctime(last_modified)}')

      st.dataframe(data.head(15), height=150)

    return data

  except FileNotFoundError:
    st.markdown(
        '### Make sure you uploaded a .csv file to 📂 File Upload. \n ### Check'
        ' the 📜 FAQ for proposed file schema.'
    )

    return 'No file uploaded yet.'


# Step 02 :: logic for data
def correlation_plots():
  """Produces correlation plot with covariate relationship indexes.
  """
  data = load_file().set_index('date')
  axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 6), dpi=120)

  for i, ax in enumerate(axes.flatten()):
    df1 = data[data.columns[i]]
    ax.plot(df1, color='red', linewidth=1)
    # Decorations
    ax.set_title(data.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)

  plt.tight_layout()
  plt.savefig('plots.png', dpi=300)
  st.image('plots.png')


def model_run(graph_out=1):
  """Runs the model and generates a plot.

  Args:
    graph_out: An integer indicating which graph to generate. 0 generates the
      plot, 1 generates the impact summary, and 2 generates the impact summary
      as a string.

  Returns:
    The impact summary.
  """
  pre_period = [pre_intervention_start, pre_intervention_end]
  post_period = [observation_start, observation_end]

  impact = CausalImpact(df, pre_period, post_period)
  impact.run()

  if graph_out == 0:
    plot = df.plot()
    plot.axvline(x=observation_start, linestyle='dashed', color='red')

  if graph_out == 1:
    impact.summary()
    impact.plot(fname='model_plot.png')
    st.image('model_plot.png')
  if graph_out == 2:
    return impact.summary('summary')


df = load_file(message=True)

vis_engine = st.selectbox(
    'Select Visualization',
    ['⚡️ Power Chart', 'Hide'],
    key='vis_start',
)

if st.session_state['vis_start'] == '⚡️ Power Chart':
  st.line_chart(df)
if st.session_state['vis_start'] == 'Hide':
  st.markdown('`Visualization Hidden by user`')

# Step 03 :: introduce logic for model creation
dates_col = pd.Series(pd.to_datetime(df.index).unique())

observation_start = st.date_input(
    'Media Launch', value=dates_col[0 + 32], key='media_launch'
)
observation_end = st.date_input(
    'Media End', value=dates_col[0 + 90], key='media_end'
)

intervention_toggle = st.checkbox(
    'Override Intervention Period', key='interv_bool'
)
timeseries_bool = st.checkbox('Timeseries Data', key='timeseries_bool')

if intervention_toggle:
  st.markdown(
      '#### ⚠️ Warning: while overriding the proposed intervention period, it'
      ' is advisable that you read the FAQ for documentation on the statsitical'
      ' implications before making your assumptions.'
  )

  pre_intervention_start = st.select_slider(
      'Pre-intervention start',
      options=dates_col,
      value=dates_col.min(),
      key='interv_start',
  )

  pre_intervention_end = st.select_slider(
      'Pre-intervention end',
      options=dates_col,
      value=observation_start - dt.timedelta(days=1),
  )

else:
  pre_intervention_start = dates_col.min()
  pre_intervention_end = observation_start - dt.timedelta(days=1)

if timeseries_bool:
  st.spinner('Building model')
  # def model_run(pre_st, pre_end, obs_st, obs_end, graph_out=1)
  model_run(1)
  model_run(2)
  summary_table = pd.read_csv('report.csv', header=0, index_col=0)
  st.dataframe(summary_table)

refresh = st.button('Summary')
if refresh:
  model_run(2)
  summary_table = pd.read_csv('report.csv', header=0, index_col=0)
  st.dataframe(summary_table)
