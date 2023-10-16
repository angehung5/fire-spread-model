#!/bin/sh

pyinstaller -F --hidden-import="sklearn" --workpath="./" --distpath="../" ../src/fire_model.py
