#!/bin/bash

less $(find /work1/patmjen/meshfit/experiments/$1/batch_output/ -type f -name *.err | sort --reverse | head -n1)
