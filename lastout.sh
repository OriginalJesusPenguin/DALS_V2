#!/bin/bash

less $(find /work1/patmjen/meshfit/experiments/$1/batch_output/ -type f -name *.out | sort --reverse | head -n1)
