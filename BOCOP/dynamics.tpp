// Function for the dynamics of the problem
// dy/dt = dynamics(y,u,z,p)

// The following are the input and output available variables
// for the dynamics of your optimal control problem.

// Input :
// time : current time (t)
// normalized_time: t renormalized in [0,1]
// initial_time : time value on the first discretization point
// final_time : time value on the last discretization point
// dim_* is the dimension of next vector in the declaration
// state : vector of state variables
// control : vector of control variables
// algebraicvars : vector of algebraic variables
// optimvars : vector of optimization parameters
// constants : vector of constants

// Output :
// state_dynamics : vector giving the expression of the dynamic of each state variable.

// The functions of your problem have to be written in C++ code
// Remember that the vectors numbering in C++ starts from 0
// (ex: the first component of the vector state is state[0])

// Tdouble variables correspond to values that can change during optimization:
// states, controls, algebraic variables and optimization parameters.
// Values that remain constant during optimization use standard types (double, int, ...).

#include "header_dynamics"
{
	// Simplified two patch dynamics

	Tdouble SB = state[0];
	Tdouble IB = state[1];
	Tdouble SV = state[2];
	Tdouble IV = state[3];

	double external = constants[0];
	double infection = constants[1];
	double coupling = constants[2];
	double treatment = constants[3];
	double NB = constants[4];
	double NV = constants[5];
	double density_strength = constants[6];

	state_dynamics[0] = - external*SB - infection*SB*IB/pow(NB, 1.0-density_strength)
	 										- coupling*infection*SB*IV/pow(NV, 1.0-density_strength);

	state_dynamics[1] = external*SB + infection*SB*IB/pow(NB, 1.0-density_strength)
	 										+ coupling*infection*SB*IV/pow(NV, 1.0-density_strength)
											- (1 + treatment*control[0])*IB;

	state_dynamics[2] = - infection*SV*IV/pow(NV, 1.0-density_strength)
	 										- coupling*infection*SV*IB/pow(NB, 1.0-density_strength);

	state_dynamics[3] = infection*SV*IV/pow(NV, 1.0-density_strength)
	 										+ coupling*infection*SV*IB/pow(NB, 1.0-density_strength)
											- (1 + treatment*control[1])*IV;

}
