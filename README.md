# Patch Optimal Control
Optimal disease control using a patch-based epidemic model

This project is provided as supplementary material to Bussell & Cunniffe 2021 (https://www.biorxiv.org/content/10.1101/2021.09.10.459742v1): *Optimal strategies to protect a sub-population at risk due to an established epidemic*.

We provide an implementation of the patch model in that paper, with code using BOCOP (http://www.bocop.org/) to optimise time-dependent disease control strategies.

## Prerequisites
The code makes use of the BOCOP package for optimisation, as well as Python. We have included a Dockerfile in this project to allow simple replication of the environment. Please refer to this for BOCOP installation steps, or http://www.bocop.org/.
