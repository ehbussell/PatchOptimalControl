"""
SIR patch model class, calling FBSM control optimisation.

Model consists of two patches with coupling between patches.
Region 2 represents the buffer region (B), and 3 the high value region (V).
There is a fixed external force of infection into region 2.
Force of Infection is frequency or density dependent.

All times are given in multiples of the infectious period, which is set
equal to one unit.

Possible controls are treatment in each patch.

States given by:    [ SB,   IB,   SV,   IV   ]
Co-states given by: [ L_SB, L_IB, L_SV, L_IV ]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from patch_model import fbsm as FBSM
from patch_model import bocop_utils
import subprocess
import os
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()


def no_control_policy(t, X):
    """
    Policy carrying out no disease management.

    Can be used with the Patch_Simplified_Model POLICY run mode.
    """

    return [0, 0]


class PatchModel:
    """
    Class to implement a 2 patch SIR model.

    Initialisation requires the following dictionary of parameters:
        'inf_rate':         Infection Rate,
        'coupling':         Coupling betwen patches,
        'ext_foi':          External F.O.I.
        'control_rate':     Control rate,
        'N_individuals':    Number of individuals ([NB,NV]),
        'state_init':       Initial state,
        'times':            Times to solve for,
        'control_init':     Initial control estimate,
        'max_budget_rate':  Maximum expenditure rate,
        'costate_final':    Terminal co-state values (defining the objective),
        'precision':        Precision to use for Hamiltonian (number of d.p.)

    Optionally, define transmission dependence as "DENSITY" or "FREQUENCY".
    """

    def __init__(self, params, dependence="DENSITY"):
        self.required_keys = ['inf_rate', 'coupling', 'ext_foi',
                              'control_rate', 'N_individuals', 'state_init',
                              'times', 'control_init', 'max_budget_rate',
                              'costate_final', 'precision']

        for key in self.required_keys:
            if key not in params:
                raise KeyError("Parameter {0} not found!".format(key))

        self.params = {k: params[k] for k in self.required_keys}
        if 'ext_foi_growth' in params:
            self.params['ext_foi_growth'] = params['ext_foi_growth']

        if dependence == "DENSITY" or dependence == "FREQUENCY":
            self.dependence = dependence
        else:
            raise ValueError("Invalid Transmission Dependence!")

        for key in params:
            if key not in self.required_keys:
                warnings.warn("Unused parameter: {0}".format(key))
        
        self.optimised_control = None

    def __repr__(self):
        retStr = "<" + self.__class__.__name__ + "\n\n"
        retStr += "Transmission dependence: {0}\n\n".format(self.dependence)
        for key in self.required_keys:
            retStr += key + ": " + str(self.params[key]) + "\n"
        retStr += ">"

        return retStr

    def print_msg(self, msg):
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))

    def state_deriv(self, t, X, U):
        """Return state derivative for 2 patch model."""
        control = U(t)
        SB, IB, SV, IV = X

        ext_foi = self.params['ext_foi']
        if 'ext_foi_growth' in self.params:
            ext_foi = self.params['ext_foi'] * np.exp(self.params['ext_foi_growth']*t)

        if self.dependence == "DENSITY":

            dSB = (-ext_foi*SB -
                   self.params['coupling']*self.params['inf_rate']*IV*SB -
                   self.params['inf_rate']*IB*SB)

            dIB = (ext_foi*SB +
                   self.params['coupling']*self.params['inf_rate']*IV*SB +
                   self.params['inf_rate']*IB*SB -
                   (1.0 + self.params['control_rate']*control[0])*IB)

            dSV = (-self.params['coupling']*self.params['inf_rate']*IB*SV -
                   self.params['inf_rate']*IV*SV)

            dIV = (self.params['coupling']*self.params['inf_rate']*IB*SV +
                   self.params['inf_rate']*IV*SV -
                   (1.0 + self.params['control_rate']*control[1])*IV)

        elif self.dependence == "FREQUENCY":

            dSB = (
                -ext_foi*SB - (
                    self.params['coupling']*self.params['inf_rate']*IV*SB /
                    self.params['N_individuals'][1]) - (
                    self.params['inf_rate']*IB*SB /
                    self.params['N_individuals'][0]))

            dIB = (
                ext_foi*SB + (
                    self.params['coupling']*self.params['inf_rate']*IV*SB /
                    self.params['N_individuals'][1]) + (
                    self.params['inf_rate']*IB*SB /
                    self.params['N_individuals'][0]) -
                (1.0 + self.params['control_rate']*control[0])*IB)

            dSV = (
                -self.params['coupling']*self.params['inf_rate']*IB*SV /
                self.params['N_individuals'][0]) - (
                self.params['inf_rate']*IV*SV /
                self.params['N_individuals'][1])

            dIV = (
                self.params['coupling']*self.params['inf_rate']*IB*SV /
                self.params['N_individuals'][0]) + (
                self.params['inf_rate']*IV*SV /
                self.params['N_individuals'][1]) - (
                1.0 + self.params['control_rate']*control[1])*IV

        else:
            raise ValueError("Invalid transmission dependence!")

        dX = [dSB, dIB, dSV, dIV]

        return dX

    def costate_deriv(self, t, L, X, U):
        """Return co-state derivative for 2 patch model."""

        control = U(t)
        SB, IB, SV, IV = X(t)
        L_SB, L_IB, L_SV, L_IV = L
        NB, NV = self.params['N_individuals']

        if self.dependence == "DENSITY":

            dL_SB = (L_SB*(
                self.params['ext_foi'] +
                self.params['inf_rate']*(self.params['coupling']*IV + IB)) -
                     L_IB*(
                self.params['ext_foi'] +
                self.params['inf_rate']*(self.params['coupling']*IV + IB)))

            dL_IB = (L_SB*self.params['inf_rate']*SB +
                     L_IB*(1.0 + self.params['control_rate']*control[0] -
                           self.params['inf_rate']*SB) +
                     L_SV*self.params['coupling']*self.params['inf_rate']*SV -
                     L_IV*self.params['coupling']*self.params['inf_rate']*SV)

            dL_SV = (L_SV*self.params['inf_rate']*(
                        self.params['coupling']*IB + IV) -
                     L_IV*self.params['inf_rate']*(
                        self.params['coupling']*IB + IV))

            dL_IV = (L_SV*self.params['inf_rate']*SV +
                     L_IV*(1.0 + self.params['control_rate']*control[1] -
                           self.params['inf_rate']*SV) +
                     L_SB*self.params['coupling']*self.params['inf_rate']*SB -
                     L_IB*self.params['coupling']*self.params['inf_rate']*SB)

        if self.dependence == "FREQUENCY":

            dL_SB = (L_SB*(
                self.params['ext_foi'] +
                self.params['inf_rate']*(
                    self.params['coupling']*IV/NV + IB/NB)) -
                     L_IB*(
                self.params['ext_foi'] +
                self.params['inf_rate']*(
                    self.params['coupling']*IV/NV + IB/NB)))

            dL_IB = (
                L_SB*self.params['inf_rate']*SB/NB +
                L_IB*(1.0 + self.params['control_rate']*control[0] -
                      self.params['inf_rate']*SB/NB) +
                L_SV*self.params['coupling']*self.params['inf_rate']*SV/NB -
                L_IV*self.params['coupling']*self.params['inf_rate']*SV/NB)

            dL_SV = (L_SV*self.params['inf_rate']*(
                        self.params['coupling']*IB/NB + IV/NV) -
                     L_IV*self.params['inf_rate']*(
                        self.params['coupling']*IB/NB + IV/NV))

            dL_IV = (
                L_SV*self.params['inf_rate']*SV/NV +
                L_IV*(1.0 + self.params['control_rate']*control[1] -
                      self.params['inf_rate']*SV/NV) +
                L_SB*self.params['coupling']*self.params['inf_rate']*SB/NV -
                L_IB*self.params['coupling']*self.params['inf_rate']*SB/NV)

        dL = [dL_SB, dL_IB, dL_SV, dL_IV]

        return dL

    def optimise_control(self, X, L, t):
        """Return optimal control given state and co-state."""
        SB, IB, SV, IV = X(t)
        L_SB, L_IB, L_SV, L_IV = L(t)

        IB = np.max([0, IB])
        IV = np.max([0, IV])

        # Round coefficient values to avoid convergence issues
        #   using precision value
        coef = np.around([
            -L_IB*self.params['control_rate']*IB,
            -L_IV*self.params['control_rate']*IV], self.params['precision'])
        A_ub = np.array([[IB, IV]])
        b_ub = [self.params['max_budget_rate']]
        bounds = [(0, 1), (0, 1)]

        # Set tolerance stricter than rounding
        control = linprog(coef, A_ub, b_ub, bounds=bounds,
                          options={'tol': 10**-(self.params['precision'] + 1)})

        # Find cases with coefficient of zero
        zero_ind = np.where(coef == 0)[0]
        inf_state = np.around([IB, IV], self.params['precision'])
        inf_costate = np.around([L_IB, L_IV], self.params['precision'])
        control.x[zero_ind] = 0

        # Sort by decreasing infection costate value to correctly prioritise
        SORTED_zero_ind = zero_ind[np.argsort(inf_costate[zero_ind])[::-1]]

        for ind in SORTED_zero_ind:
            if inf_costate[ind] > 0:
                # Here positive costate implies control should be maximal
                #   so spend as much budget as possible
                control.x[ind] = np.max([
                    0, np.min([1, np.nan_to_num(np.divide(
                        self.params['max_budget_rate'] -
                        np.sum(control.x*inf_state),
                        inf_state[ind]))])])

        return np.array(control.x)

    def run(self, mode="FBSM", *args, **kwargs):
        """
        Run model choosing which mode to operate in.

        Options for operating mode are:
        FBSM    -   optimise using forward-backward sweep
        BOCOP   -   optimise using BOCOP direct optimisation
        POLICY  -   run a particular control policy that depends on
                    state and time
        MPC     -   run using model predictive control
        """

        if mode == "FBSM":
            return self.run_fbsm(*args, **kwargs)
        elif mode == "BOCOP":
            return self.run_BOCOP(*args, **kwargs)
        elif mode == "POLICY":
            return self.run_policy(*args, **kwargs)
        elif mode == "MPC":
            return self.run_MPC(*args, **kwargs)
        else:
            raise ValueError("Invalid run mode!")

    def run_fbsm(self, verbose=True, plots=True, maxIters=None, tol=0.001):
        """Run FBSM solver and return optimal state, co-state and control."""

        self._mode = "FBSM"
        # Ignore divide by zero errors for update step
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        args = [self.state_deriv, self.costate_deriv,
                self.params['state_init'], self.params['costate_final'],
                self.params['control_init'], self.params['times'],
                self.optimise_control]

        # Set up FBSM solver
        self.Solver = FBSM.FBSolver(*args)

        self.Solver.tol = tol
        self.Solver.verbose = verbose
        if plots:
            self.Solver.show_plot(10)

        Xt, Lt, Ut = self.Solver.runOptimisation(maxIters=maxIters)

        expense_regionB = np.array([
            Ut(t)[0]*Xt(t)[1] for t in self.params['times']])
        expense_regionV = np.array([
            Ut(t)[1]*Xt(t)[3] for t in self.params['times']])

        self.print_msg("Maximum Expenditure:  {0}".format(np.max(
            expense_regionB + expense_regionV)))

        np.seterr(**old_err_settings)
        self._mode = None

        return (Xt, Lt, Ut)

    def run_policy(self, control_policy):
        """Run forward simulation using a given control policy.

        Function control_policy(t, X) returns list of budget allocations for
            each region - [budget_buffer, budget_value]
        """

        self._mode = "POLICY"
        self._controller = control_policy

        # Ignore divide by zero errors when calculating control
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        def state_deriv_policy(t, X):
            # fB, fV = self._get_control_from_budget(t, X)
            fB, fV = control_policy(t, X)

            def U(t):
                return [fB, fV]

            return self.state_deriv(t, X, U)

        ode = integrate.ode(state_deriv_policy)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(
            self.params['state_init'], self.params['times'][0])

        ts = [self.params['times'][0]]
        xs = [self.params['state_init']]

        for time in self.params['times'][1:]:
            ode.integrate(time)
            ts.append(ode.t)
            xs.append(ode.y)

        timespan = np.vstack(ts)
        X = np.vstack(xs).T

        Xt = interp1d(self.params['times'], X, fill_value="extrapolate")

        U = np.array([self._get_control_from_budget(t, Xt(t))
                      for t in self.params['times']]).T
        Ut = interp1d(self.params['times'], U, fill_value="extrapolate")

        np.seterr(**old_err_settings)
        self._mode = None

        return (Xt, Ut)

    def run_BOCOP(self, BOCOP_dir=None, verbose=True):
        """Run BOCOP solver and return optimal state, co-state and control."""

        self._mode = "BOCOP"

        if BOCOP_dir is None:
            BOCOP_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "BOCOP"
            )
        
        bocop_exe = os.path.join(BOCOP_dir, "bocop.exe")
        sol_file = os.path.join(BOCOP_dir, "problem.sol")

        set_BOCOP_params(self.params, self.dependence, folder=BOCOP_dir)

        if verbose is True:
            subprocess.run([bocop_exe], cwd=BOCOP_dir)
        else:
            subprocess.run([bocop_exe],
                           cwd=BOCOP_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        Xt, Lt, Ut, exit_text = bocop_utils.readSolFile(sol_file)

        self._mode = None

        self.optimised_control = (Xt, Lt, Ut, exit_text)

        return (Xt, Lt, Ut, exit_text)

    def run_MPC(self, optimiser, MPCperiod=None, verbose=True):
        """Run simulation using optimiser to predict optimal control.

        Function optimiser(t, X) returns predicted future budget allocation as
            an interpolated function Bt.
        The MPCperiod value sets how often the control should be updated.  A
            value of None gives open-loop control.
        """

        self._mode = "MPC"

        # Ignore divide by zero errors when calculating control
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        def state_deriv_policy(t, X):
            fB, fV = self._get_control_from_budget(t, X)

            return self.state_deriv(t, X, lambda t: [fB, fV])

        time_iter = iter(self.params['times'][1:])
        time = next(time_iter)
        if MPCperiod is not None:
            updateTime = self.params['times'][0] + MPCperiod
        else:
            MPCperiod = np.inf
            updateTime = MPCperiod

        self._controller = optimiser(
            self.params['times'][0], self.params['state_init'])

        ts = [self.params['times'][0]]
        xs = [self.params['state_init']]
        us = [self._get_control_from_budget(ts[0], xs[0])]

        ode = integrate.ode(state_deriv_policy)
        ode.set_integrator('dopri5', nsteps=1000)
        ode.set_initial_value(
            self.params['state_init'], self.params['times'][0])

        while time is not None:
            while time is not None and time < updateTime:
                if verbose is True:
                    self.print_msg("Advance to time " + str(time))
                ode.integrate(time)
                ts.append(ode.t)
                xs.append(ode.y)
                us.append(self._get_control_from_budget(ts[-1], xs[-1]))
                time = next(time_iter, None)
            if time is None:
                pass
            else:
                if verbose is True:
                    self.print_msg("Advance to update time " + str(updateTime))
                ode.integrate(updateTime)
                ts.append(ode.t)
                xs.append(ode.y)
                us.append(self._get_control_from_budget(ts[-1], xs[-1]))
                if verbose is True:
                    self.print_msg("Update Controller")
                self._controller = optimiser(ts[-1], xs[-1])
                ts.append(ode.t)
                xs.append(ode.y)
                us.append(self._get_control_from_budget(ts[-1], xs[-1]))
                if np.isclose(time, updateTime):
                    time = next(time_iter, None)
                updateTime = updateTime + MPCperiod

        X = np.vstack(xs).T

        Xt = interp1d(ts, X, fill_value="extrapolate")

        U = np.vstack(us).T
        Ut = interp1d(ts, U, fill_value="extrapolate")

        np.seterr(**old_err_settings)
        self._mode = None

        return (Xt, Ut)

    def _get_control_from_budget(self, time, state):
        """Return control parameters from budget allocation."""

        if self._mode == "POLICY":
            BB, BV = self._controller(time, state)
        elif self._mode == "MPC":
            BB, BV = self._controller(time)
        else:
            raise ValueError("Invalid run mode!")

        fB = np.max([0, np.min([1, np.nan_to_num(
            np.divide(BB, state[1]))])])
        fV = np.max([0, np.min([1, np.nan_to_num(
            np.divide(BV, state[3]))])])

        return [fB, fV]

    def plot_state(self, Xt, Ut, Lt=None):
        """Plot the simulation outcome for state, control and co-state."""

        expense_regionB = np.array([
            Ut(t)[0]*Xt(t)[1] for t in self.params['times']])
        expense_regionV = np.array([
            Ut(t)[1]*Xt(t)[3] for t in self.params['times']])

        if Lt is not None:
            nrow = 3
        else:
            nrow = 2

        plt.style.use("ggplot")

        plt.figure()

        plt.subplot(nrow, 2, 1)
        plt.plot(self.params['times'],
                 np.array([Xt(t) for t in self.params['times']])[:, 0],
                 label="S")
        plt.plot(self.params['times'],
                 np.array([Xt(t) for t in self.params['times']])[:, 1],
                 label="I")
        plt.plot(self.params['times'], self.params['N_individuals'][0] -
                 np.array([Xt(t) for t in self.params['times']])[:, 0] -
                 np.array([Xt(t) for t in self.params['times']])[:, 1],
                 label="R")
        plt.ylim([0, self.params['N_individuals'][0]])
        plt.legend()
        plt.title("Buffer Region - State")

        plt.subplot(nrow, 2, 2)
        plt.plot(self.params['times'],
                 np.array([Xt(t) for t in self.params['times']])[:, 2:4])
        plt.plot(self.params['times'], self.params['N_individuals'][1] -
                 np.array([Xt(t) for t in self.params['times']])[:, 2] -
                 np.array([Xt(t) for t in self.params['times']])[:, 3])
        plt.ylim([0, self.params['N_individuals'][1]])
        plt.title("High Value Region - State")

        if Lt is not None:
            plt.subplot(nrow, 2, 3)
            plt.plot(self.params['times'],
                     np.array([Lt(t) for t in self.params['times']])[:, 0],
                     label="S")
            plt.plot(self.params['times'],
                     np.array([Lt(t) for t in self.params['times']])[:, 1],
                     label="I")
            plt.legend()
            plt.title("Buffer Region - Co-State")

            plt.subplot(nrow, 2, 4)
            plt.plot(self.params['times'],
                     np.array([Lt(t) for t in self.params['times']])[:, 2:4])
            plt.title("High Value Region - Co-State")

        plt.subplot(nrow, 2, 2*nrow - 1)
        plt.plot(self.params['times'],
                 np.array([Ut(t) for t in self.params['times']])[:, 0])
        plt.ylim([-0.01, 1.01])
        plt.title("Buffer Region - Control")

        plt.subplot(nrow, 2, 2*nrow)
        plt.plot(self.params['times'],
                 np.array([Ut(t) for t in self.params['times']])[:, 1])
        plt.ylim([-0.01, 1.01])
        plt.title("High Value Region - Control")

        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.fill_between(self.params['times'], 0, expense_regionV,
                         label="High Value Region", color="Purple", alpha=0.5)
        plt.fill_between(self.params['times'], expense_regionV,
                         expense_regionB + expense_regionV,
                         label="BufferRegion", color="Green", alpha=0.5)
        plt.plot(self.params['times'],
                 np.full_like(self.params['times'],
                 self.params['max_budget_rate']), 'r--',
                 label="Max Expenditure")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Control Expenditure")
        plt.ylim([0, self.params['max_budget_rate'] * 1.1])

        plt.tight_layout()
        plt.show()


def set_BOCOP_params(params, dependence="DENSITY", folder=None):

    if folder is None:
        folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "BOCOP"
        )

    if dependence == "DENSITY":
        density_strength = 1.0
    elif dependence == "FREQUENCY":
        density_strength = 0.0
    elif 0 <= dependence <= 1:
        density_strength = dependence
    else:
        raise ValueError("Invalid transmission dependence!")

    with open(os.path.join(folder, "problem.bounds"), "r") as f:
        allLines = f.readlines()

    # Initial conditions
    allLines[7] = str(params['state_init'][0]) + " " + \
        str(params['state_init'][0]) + " equal\n"
    allLines[8] = str(params['state_init'][1]) + " " + \
        str(params['state_init'][1]) + " equal\n"
    allLines[9] = str(params['state_init'][2]) + " " + \
        str(params['state_init'][2]) + " equal\n"
    allLines[10] = str(params['state_init'][3]) + " " + \
        str(params['state_init'][3]) + " equal\n"

    # Max budget expenditure
    allLines[21] = "-2e+020 " + str(params['max_budget_rate']) + " upper\n"

    with open(os.path.join(folder, "problem.bounds"), "w") as f:
        f.writelines(allLines)

    with open(os.path.join(folder, "problem.constants"), "r") as f:
        allLines = f.readlines()

    allLines[5] = str(params['ext_foi']) + "\n"
    allLines[6] = str(params['inf_rate']) + "\n"
    allLines[7] = str(params['coupling']) + "\n"
    allLines[8] = str(params['control_rate']) + "\n"
    allLines[9] = str(params['N_individuals'][0]) + "\n"
    allLines[10] = str(params['N_individuals'][1]) + "\n"
    allLines[11] = str(density_strength) + "\n"

    with open(os.path.join(folder, "problem.constants"), "w") as f:
        f.writelines(allLines)

    with open(os.path.join(folder, "problem.def"), "r") as f:
        allLines = f.readlines()

    nSteps = str(len(params['times']) - 1)
    allLines[5] = "time.initial double " + str(params['times'][0]) + "\n"
    allLines[6] = "time.final double " + str(params['times'][-1]) + "\n"
    allLines[18] = "discretization.steps integer " + nSteps + "\n"

    with open(os.path.join(folder, "problem.def"), "w") as f:
        f.writelines(allLines)
