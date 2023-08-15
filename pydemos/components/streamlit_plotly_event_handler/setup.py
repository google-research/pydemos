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

"""Defines pip package that wraps a Streamlit component.

Declares setup for pip package by parsing information from the Streamlit
component's package.json.
"""

import json
import os

import setuptools


try:
  parent_dir = os.path.dirname(os.path.abspath(__file__))
  json_dir = os.path.join(
      parent_dir, "src/streamlit_plotly_event_handler/frontend/package.json")
  package_info = json.load(open(json_dir, "r", encoding="utf-8"))
except Exception as e:
  print("An exception has ocurred while loading the package.json file:", e)
  raise

setuptools.setup(
    name=package_info["name"],
    version=package_info["version"],
    author=package_info["author"]["name"],
    email=package_info["author"]["email"],
    description=package_info["description"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.6",
    install_requires=[
        "streamlit >= 0.63",
        "plotly >= 4.14.3",
    ],
)
