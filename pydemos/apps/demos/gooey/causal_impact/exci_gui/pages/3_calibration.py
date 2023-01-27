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

"""This script uses streamlit elements to display descriptive stats from the model.

"""

import streamlit as st

stat_sig = 0.9544 *100
lift = 0.23445 *100

st.markdown("## Causal Model Report")
st.write(f" Statistical Significance {stat_sig}%")
st.write(f" Observed Incremental Lift {lift}%")
