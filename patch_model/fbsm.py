"""Forward-Backward sweep methods for optimal control."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from collections import deque
import itertools
import time


class FBSolver:
    """
    Forward-Backward Sweep Method for solving optimal control problems.

    Attributes:
      state_deriv:    Function returning rate of change of the state variables.
                      Must have arguments (t,X,U)
                      where U(t) is a function returning the control at time t.
      state_init:     Value of state variables at initial time.
      costate_deriv:  Function returning rate of change of the costate.
                      variables.  Must have arguments (t,L,X,U) where U(t) is a
                      function returning the control at time t and X(t) is a
                      function returning the state at time t.
      costate_final:  Value of costate variables at final time T.
      control_init:   Initial guess for the optimal control at each time
                      in t_vals.
      tol:            Accepted tolerance for convergence (default: 0.001).
      t_vals:         The times to solve for the optimal control.
      control_optim:  Function returning the optimal control for given state.
                      Must have arguments (X,L,t)
                      where X(t) is a function returning the state at time t
                      and L(t) is a function returning the costate at time t.
      verbose:        Boolean giving verbosity of solver.
                      If True, the convergence value is printed at each
                      iteration.
    """

    def __init__(self, dX, dL, X0, LT, U0, t_vals, optimalU):
        self.state_deriv = dX
        self.costate_deriv = dL
        self.state_init = X0
        self.costate_final = LT
        self.t_vals = t_vals
        self.control_init = U0
        self.control_optim = optimalU
        self.tol = 0.001
        self.verbose = False
        self.bounds = None

    @property
    def state_deriv(self):
        """Get or set the state rate of change function"""
        return self._state_deriv

    @state_deriv.setter
    def state_deriv(self, state_deriv):
        self._state_deriv = state_deriv

    @property
    def state_init(self):
        """Get or set the initial state"""
        return self._state_init

    @state_init.setter
    def state_init(self, state_init):
        self._state_init = state_init

    @property
    def costate_deriv(self):
        """Get or set the costate rate of change function"""
        return self._costate_deriv

    @costate_deriv.setter
    def costate_deriv(self, costate_deriv):
        self._costate_deriv = costate_deriv

    @property
    def costate_final(self):
        """Get or set the terminal costate"""
        return self._costate_final

    @costate_final.setter
    def costate_final(self, costate_final):
        self._costate_final = costate_final

    @property
    def control_init(self):
        """Get or set the initial control guess

        Control is interpolated between time values and stored as a function
        """
        return self._control_init

    @control_init.setter
    def control_init(self, control_init):
        u_func = interp1d(self.t_vals, control_init, fill_value="extrapolate")
        self._control_init = u_func

    @property
    def tol(self):
        """Get or set the tolerance value"""
        return self._tol

    @tol.setter
    def tol(self, tol):
        self._tol = tol

    @property
    def t_vals(self):
        """Get or set the times to solve for"""
        return self._t_vals

    @t_vals.setter
    def t_vals(self, t_vals):
        self._t_vals = t_vals

    @property
    def control_optim(self):
        """Get or set the optimal control function"""
        return self._control_optim

    @control_optim.setter
    def control_optim(self, control_optim):
        self._control_optim = control_optim

    @property
    def verbose(self):
        """Get or set verbosity"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbosity):
        self._verbose = bool(verbosity)

    @property
    def bounds(self):
        """Get or set control bounds"""
        return self._bounds

    @bounds.setter
    def bounds(self, control_bounds):
        self._bounds = control_bounds
        if hasattr(self, "_interactive_plot"):
            self._interactive_plot.bounds = self._bounds

    def print_msg(self, msg):
        """Print message, indicating class message originated from"""
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))

    def show_plot(self, plot_last_N):
        """Set the simulation to interactively plot the updated controls"""

        self._interactive_plot = InteractivePlot(
            plot_last_N, self.t_vals, self.control_init, self.verbose, self.bounds)

    def forward(self, control):
        """Run forward simulation of state using given control"""

        if self.verbose:
            self.print_msg("Forward simulation")

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(self.state_init, self.t_vals[0])
        ode.set_f_params(control)

        ts = [self.t_vals[0]]
        xs = [self.state_init]

        for time in self.t_vals[1:]:
            ode.integrate(time)
            ts.append(ode.t)
            xs.append(ode.y)

        timespan = np.vstack(ts)
        X = np.vstack(xs).T

        Xfunc = interp1d(self.t_vals, X, fill_value="extrapolate")

        return Xfunc

    def backward(self, state, control):
        """Run backward simulation of costate using given control and state"""

        if self.verbose:
            self.print_msg("Backward simulation")

        ode = integrate.ode(self.costate_deriv)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(self.costate_final, self.t_vals[-1])
        ode.set_f_params(state, control)

        ts = [self.t_vals[-1]]
        xs = [self.costate_final]

        for time in self.t_vals[-2::-1]:
            ode.integrate(time)
            ts.append(ode.t)
            xs.append(ode.y)

        timespan = np.vstack(ts)
        L = np.vstack(xs).T[:, ::-1]

        Lfunc = interp1d(self.t_vals, L, fill_value="extrapolate")

        return Lfunc

    def updateU(self, control, state, costate):
        """Update the control values for new state and costate"""

        if self.verbose:
            self.print_msg("Update controls")

        U_old = np.array([control(t) for t in self.t_vals]).T
        U_star = np.array([self.control_optim(state, costate, t)
                           for t in self.t_vals]).T

        U_new = (U_old + U_star) / 2

        U_newFunc = interp1d(self.t_vals, U_new, fill_value="extrapolate")

        return U_newFunc

    def checkConvergence(self, Xold, Xnew, Lold, Lnew, Uold, Unew):
        """Check whether the system has converged to the accepted tolerance.

        Tolerance requirement is given by:
            tol*||X|| - ||X - old_X|| >= 0
        for state, costate and control variables
        """

        x_old = np.array([Xold(t) for t in self.t_vals])
        x_new = np.array([Xnew(t) for t in self.t_vals])
        l_old = np.array([Lold(t) for t in self.t_vals])
        l_new = np.array([Lnew(t) for t in self.t_vals])
        u_old = np.array([Uold(t) for t in self.t_vals])
        u_new = np.array([Unew(t) for t in self.t_vals])

        tmp_x = (self.tol * np.sum(np.absolute(x_new), axis=1) -
                 np.sum(np.absolute(x_new - x_old), axis=1))
        tmp_l = (self.tol * np.sum(np.absolute(l_new), axis=1) -
                 np.sum(np.absolute(l_new - l_old), axis=1))
        tmp_u = (self.tol * np.sum(np.absolute(u_new), axis=1) -
                 np.sum(np.absolute(u_new - u_old), axis=1))

        test = np.min(np.concatenate([tmp_x, tmp_l, tmp_u]))

        if self.verbose:
            self.print_msg("Convergence test: " + str(test))

        if test < 0:
            return False
        else:
            return True

    def runOptimisation(self, maxIters=None):
        """Optimise controls using the Forward-Backward Sweep Method"""

        if hasattr(self, "_interactive_plot"):
            self._interactive_plot.plot()

        self.print_msg("Optimising...")
        start_time = time.time()

        try:

            converged = False
            Xnew = self.forward(self.control_init)
            Lnew = self.backward(Xnew, self.control_init)
            Unew = self.updateU(self.control_init, Xnew, Lnew)

            if hasattr(self, "_interactive_plot"):
                self._interactive_plot.update_plot(Unew)

            iters = 0

            while converged is False:
                Xold = Xnew
                Lold = Lnew
                Uold = Unew

                Xnew = self.forward(Uold)
                Lnew = self.backward(Xnew, Uold)
                Unew = self.updateU(Uold, Xnew, Lnew)

                if hasattr(self, "_interactive_plot"):
                    self._interactive_plot.update_plot(Unew)

                converged = self.checkConvergence(Xold, Xnew, Lold, Lnew,
                                                Uold, Unew)
                iters = iters + 1

                if (iters % 100 == 0):
                    self.print_msg("...{0} iterations complete...".format(iters))

                # Check for max iterations

            if maxIters is not None:
                if iters >= maxIters:
                    raise StopIteration
        except (KeyboardInterrupt, StopIteration):
            end_time = time.time()
            tot_time = end_time - start_time

            self.print_msg("Optimisation cancelled after {0} iterations"
                            .format(iters))
            self.print_msg(
                "Time taken: {0:.1f} seconds".format(tot_time))

            Xnew = self.forward(Unew)
            Lnew = self.backward(Xnew, Unew)

            if hasattr(self, "_interactive_plot"):
                self._interactive_plot.final()

            return (Xnew, Lnew, Unew)

        end_time = time.time()
        tot_time = end_time - start_time

        self.print_msg("Converged in {0} iterations".format(iters))
        self.print_msg("Time taken: {0:.1f} seconds".format(tot_time))

        Xnew = self.forward(Unew)
        Lnew = self.backward(Xnew, Unew)

        if hasattr(self, "_interactive_plot"):
            self._interactive_plot.final()

        return (Xnew, Lnew, Unew)


class InteractivePlot:
    """Implement interactive plot during FBSM optimisation.

    Class plots the current updated control functions and those from previous
    iterations, back to a maximum specified number.
    """

    def __init__(self, plot_last_N, t_vals, control_init, verbose, bounds=None):
        if verbose:
            self.print_msg("Initialising InteractivePlot class")

        self.lastN = plot_last_N
        self.queue = deque([control_init], maxlen=self.lastN)
        self.t_vals = t_vals
        self.verbose = verbose
        self.alphas = np.linspace(0, 1, self.lastN+1)[-2:0:-1]
        self.num_controls = len(np.atleast_1d(control_init(0)))

        # Set up plot environment
        plt.ion()
        plt.style.use("ggplot")
        self.fig = plt.figure()
        self.ax = []
        for control in range(self.num_controls):
            self.ax.append(self.fig.add_subplot(
                "1" + str(self.num_controls) + str(control+1)))
            self.ax[control].set_xlim([min(self.t_vals), max(self.t_vals)])
            self.ax[control].set_title("Control {0}".format(control + 1))

        self.lines = [[None]*self.lastN for x in range(self.num_controls)]

        plt.tight_layout()

        if bounds is None:
            self.bounds = [[0, 1]] * self.num_controls
        else:
            self.bounds = bounds

    @property
    def bounds(self):
        """Get or set the control bounds."""
        return self._bounds

    @bounds.setter
    def bounds(self, control_bounds):
        self._bounds = control_bounds
        for control in range(self.num_controls):
            if self._bounds is None:
                self.ax[control].set_ylim([-0.01, 1.01])
            else:
                self.ax[control].set_ylim([self._bounds[control][0] - 0.01,
                                           self._bounds[control][1] + 0.01])

    def plot(self):
        """Plot the current list of lines"""
        if self.verbose:
            self.print_msg("Drawing plot")

        for control in range(self.num_controls):
            # Remove previously drawn lines
            for line in [x for x in self.lines[control] if x is not None]:
                line.remove()

            # Plot previous iterations in blue, increasing transparency
            #   for older cases
            prev_iters = enumerate(reversed(
                    list(itertools.islice(self.queue, len(self.queue)-1))))
            for idx, val in prev_iters:
                self.lines[control][idx], = self.ax[control].plot(
                    self.t_vals, np.array([
                        np.atleast_1d(val(t))
                        for t in self.t_vals])[:, control],
                    'b', alpha=self.alphas[idx])

            # Plot most recent controls in red
            self.lines[control][-1], = self.ax[control].plot(
                self.t_vals, np.array([
                    np.atleast_1d(self.queue[-1](t))
                    for t in self.t_vals])[:, control], 'r')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, new_control):
        """Update plot with most recent control functions"""
        if self.verbose:
            self.print_msg("Updating plot")

        self.queue.append(new_control)
        self.plot()

    def final(self):
        """Stop interactive plotting"""
        plt.ioff()

    def print_msg(self, msg):
        """Print message, indicating class message originated from"""
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))
