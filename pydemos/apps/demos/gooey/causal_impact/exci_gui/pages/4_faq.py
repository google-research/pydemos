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

import streamlit as st

st.markdown('### Causal Impact GUI')
st.markdown(
    '##### Workstream and GUI Authors \n sebpf@, ethanmarkowitz@, charliemancuso@, mattwooten@'
)

st.markdown("""
    # Causal Impact Model Best Practices
    ### Data Requirements:
    ##### A feature to test your client on
    - Client-provided - `Conversion Action`
    - WildcatX / Google Ads - `Clicks`
    - WildcatX / Google Ads - `Impressions`
    ##### Features for 5 or more peers
    - Equivalent of `Conversion Action` , `Clicks`, or `Impressions` for 5 or more peers as **separate features** .
        Toggle *Peer Breakdown* in WildcatX, for example.
    ##### Date Granularity
    - You will need **at least** 3 months worth of data.
        ###### Rule of thumb is add 1 additional month to the original 3 month requirement for every 2 weeks of conversion lag your action takes above 2 weeks.
        ## FAQ 🧑‍💻
        # - *Why can't I load the required dependencies manually?*
        # Due to the compatibility needs of  GUI with the CausalImpact library.
        # **tl;dr** - a ton of modifications had to be made to the original source code. Specifically files `analysis.py` and `misc.py` were impacted in order to produce richer information in the visual interface itself, contrary to the backend outputs produced originally by CausalImapct 0.2.4.
        """)
