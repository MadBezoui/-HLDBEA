#!/bin/bash

#Execute the following command to reinstall the venv when upgrading Ubuntu to a new version:
if [[ "$1" == "v" ]] ; then
    python3 -m venv .pymoo-venv
    exit
fi

source ./.pymoo-venv/bin/activate

if [[ "$1" == "i" ]] ; then
    pip install -U pymoo
    pip install -U pyrecorder
    pip install pandas
    pip install seaborn
    pip install numba
    pip install pyyaml
    #Optional(?)
    pip install textual textual-dev
elif [[ "$1" == "u" ]] ; then
    pip install -U pymoo
elif [[ "$1" == "t" ]] ; then
    ./.pymoo-venv/bin/python ./textual_gui_launcher.py
else
    #./.pymoo-venv/bin/python ./ibea_test.py
    #./.pymoo-venv/bin/python ./nibea_test.py
    #./.pymoo-venv/bin/python ./nibea_test.py --profile configs/profile_debug_final.yaml --params configs/params_standard.yaml
    ./.pymoo-venv/bin/python ./nibea_test.py --profile configs/profile_debug.yaml --params configs/params_standard.yaml
fi

deactivate
