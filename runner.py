#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Just a little script to run all the possible combinations.

import os

#run each of my specifications
sizes = [64, 128, 256, 512]
for thisSize in sizes:
  for batchSize in range(1, 17):
    commandStr = "python3 my-unet.py 50000 " + str(thisSize) + " 100 " + str(batchSize)
    print("Running: " + commandStr)
    os.system(commandStr)
