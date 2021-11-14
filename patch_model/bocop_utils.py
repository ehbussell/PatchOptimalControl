"""Functions to help when using the BOCOP direct solver for optimal control
problems."""

import logging
import os
import numpy as np
from scipy.interpolate import interp1d


def readSolFile(file="problem.sol", ignore_fail=False):
    """Read BOCOP solution file and extract Xt, Lt and Ut as interpolated functions."""

    with open(file, 'r') as f:
        allLines = f.readlines()

    for i, line in enumerate(allLines):
        if "time.initial" in line:
            time_init = float(line.split()[-1])
        if "time.final" in line:
            time_final = float(line.split()[-1])
        if "state.dimension" in line:
            state_dim = int(line.split()[-1])
        if "control.dimension" in line:
            control_dim = int(line.split()[-1])
        if "discretization.steps" in line:
            time_steps = int(line.split()[-1])

    times = np.linspace(time_init, time_final, time_steps + 1)

    Xt = []
    Lt = []
    Ut = []

    for i, line in enumerate(allLines):
        for j in range(state_dim):
            if "# State " + str(j) + "\n" == line:
                Xt.append(list(map(float, allLines[i+1:i+2+time_steps])))

        for j in range(control_dim):
            if "# Control " + str(j) + "\n" == line:
                Ut.append(list(map(float, allLines[i+1:i+1+2*time_steps])))

        for j in range(state_dim):
            if ("# Dynamic constraint " + str(j)) in line:
                Lt.append(list(map(float, allLines[i+2:i+2+time_steps])))

    results_file = os.path.join(os.path.dirname(file), "result.out")
    with open(results_file, "r") as infile:
        result_lines = infile.readlines()

    exit_text = None
    for line in result_lines:
        if "EXIT" in line:
            exit_text = line[6:].strip()

    if (exit_text != "Optimal Solution Found."
            and exit_text != "Solved To Acceptable Level.") and not ignore_fail:
        raise RuntimeError("BOCOP optimisation failed with code: {0}".format(exit_text))

    if exit_text == "Some uncaught Ipopt exception encountered.":
        logging.error("Uncaught Ipopt exception. Try re-running optimiser.")
        return (None, None, None, exit_text)

    Xt = interp1d(times, Xt, fill_value="extrapolate")
    Lt = interp1d(times[:-1], Lt, fill_value="extrapolate")
    Ut_mean = np.mean(np.reshape(np.array(Ut), (control_dim, -1, 2)), axis=2)
    Ut = interp1d(times[:-1], Ut_mean, fill_value="extrapolate")

    return (Xt, Lt, Ut, exit_text)
