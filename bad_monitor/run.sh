#!/bin/bash
python3 controller_sim.py | streamlab-linux monitor spec_linear_controller.lola --stdout --online --verbosity outputs > monitor_out.txt 


