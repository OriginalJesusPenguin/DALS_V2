#!/bin/bash

less $(find /work1/patmjen/meshfit/experiments/mesh_decoder/batch_output/ -type f -name *.err | sort --reverse | head -n1)
