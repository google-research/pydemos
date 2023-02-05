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

# Import dependencies
import pandas as pd
import streamlit as st
# GUI elements start loading
st.title('📂 File Upload')
st.markdown('#### Step 1. Please upload your file.')

# Step 02 :: import data
data_file = st.file_uploader('Choose a file.')


# Step 03 :: clean and encode data
def column_encode(data):
  """This function takes in a pandas dataframe and returns the columns selected on the streamlit GUI as Date and Test, followed by x1..xn for every other column (covariates).

  This is the necessary schema for the CausalImpact library.
  Returns: column_names
  """
  column_list = [date_selection, test_selection]
  columns = list(data.columns.values)

  for i in columns:
    if i not in column_list:
      column_list.append(i)

  column_names = column_list[:2]
  for i in range(1, len(column_list) - 1):
    column_names.append(f'x{i}')

  return column_names


# Step 04 :: write UI elements and introduce function logic
if data_file is None:
  st.markdown('##### Follow FAQ for file schema and datatype documentation.')

if data_file is not None:
  # read data and confirm receipt
  df = pd.read_csv(data_file)
  st.markdown('#### ✅ File Loaded successfully.')

  # define columns to feed dropdown
  columns = df.columns
  # write a dropdown for the selection of the date column
  date_selection = st.selectbox(
      'Please select a date column. The tool will take care of the rest',
      columns)
  # write dropdown for the test variable, default value with index=1
  test_selection = st.selectbox('Test Variable', columns, index=1)

  # if user clicks button, data stored in WD
  check = st.button('Encode 🪢')

  if check is False:
    st.text('CLEANUP_SETUP_HERE')
  if check:
    # pass dataframe through our custom column_encode function
    df.columns = column_encode(df)
    # renames date so we can read it 'blindly' at model bake
    df = df.rename(columns={date_selection: 'date', test_selection: 'y'})
    df = df.set_index(['date'])
    # dataframe receipt for user
    st.dataframe(df)
    df.to_csv('data_file.csv')
    st.markdown('### Proceed to 🚀Bake Model!')