#!/bin/bash
python3 controller_sim.py | streamlab-linux monitor spec_linear_controller.lola --online --verbosity outputs 


