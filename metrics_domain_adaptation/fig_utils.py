
#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#
# Sets misc plotting defaults.
# Should be imported in all plotting scripts even if no variable or function is being used.
#

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

FONT_MONOSPACE = {'fontname': 'monospace'}
MARKERS = "o^s*DP1"
COLORS = [
    "darkseagreen",
    "salmon",
    "cornflowerblue",
    "seagreen",
    "orange",
    "lightpink",
    "dimgray",
]
COLORS_DOMAIN = {
    "bio": "#ddd",
    "general": "#888",
}
COLORS_FIRE = ["#9c2963", "#282e9b", "#fb9e07"]

mpl.rcParams['axes.prop_cycle'] = cycler(color=COLORS)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams["hatch.linewidth"] = 5

# mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['font.family'] = "serif"


MARKER_SQUARE = {
    "bbox": {"boxstyle": f"square,pad=0.15", "fc": "#eee", "alpha": 0.9},
    "font": {"size": 8},
    "ha": "center", "va": "center",
}
MARKER_TRIANGLE = {
    "bbox": {"boxstyle": f"triangle,pad=0.15", "fc": "#eee", "alpha": 0.9},
    "font": {"size": 8},
    "ha": "center", "va": "center",
}
MARKER_SAW = {
    "bbox": {"boxstyle": f"Sawtooth,pad=0.15,tooth_size=0.15", "fc": "#eee", "alpha": 0.9},
    "font": {"size": 8},
    "ha": "center", "va": "center",
}
MARKER_CIRCLE = {
    "bbox": {"boxstyle": f"circle,pad=0.15", "fc": "#eee", "alpha": 0.9},
    "font": {"size": 8},
    "ha": "center", "va": "center",
}
MARKER_CIRCLE_2 = {
    **MARKER_CIRCLE,
    "bbox": {"boxstyle": f"circle,pad=0.2", "fc": "#eee", "alpha": 0.9},
    "font": {"size": 7},
}
MARKER_CIRCLE_DASH = {
    **MARKER_CIRCLE,
    "bbox": {"boxstyle": f"circle,pad=0.0", "fc": "#eee", "alpha": 0.9, "linestyle": (0, (1, 1))},
    "font": {"size": 10},
}


MARKER_CONF_INT = dict(
    color="#ccc",
    marker="_",
    markersize=6,
    linewidth=1,
    linestyle=":",
)

# taken from 03/01-
EXPECTED_BOOST = 0.0268


def save(name, root="./"):
    import os
    # two folders: PDF for the paper and PNG for the README
    os.makedirs(f"{root}computed/figures_pdf", exist_ok=True)
    os.makedirs(f"{root}computed/figures_png", exist_ok=True)
    # plot PNG with low res because it's in git
    plt.savefig(f"{root}computed/figures_png/{name}.png", dpi=300)
    plt.savefig(f"{root}computed/figures_pdf/{name}.pdf")


def diff_to_arrow_dy(diff):
    diff /= 5
    diff = (-1 if diff < 0 else 1)*max(0.005, abs(diff))
    return -0.01 if diff < 0 else 0.01
