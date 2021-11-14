"""Example script for optimising control on patch model."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from patch_model.patch_model import PatchModel

def make_figure2(model):
    "Replicate Figure 2 from paper - General Strategy"

    times = model.params['times']
    max_budget_rate = model.params['max_budget_rate']
    N_individuals = model.params['N_individuals']
    Xt, Lt, Ut, _ = model.optimised_control

    expense_region1 = np.array([Ut(t)[0]*Xt(t)[1] for t in times])
    expense_region2 = np.array([Ut(t)[1]*Xt(t)[3] for t in times])

    fig = plt.figure(figsize=(6.2, 3.2))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[2, 1], left=0.08, right=0.65, wspace=0.4, hspace=0.5,
        bottom=0.25, top=0.97
    )
    gs2 = gridspec.GridSpec(2, 1, left=0.78, right=0.96, bottom=0.25, top=0.97)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs2[0], sharex=ax2)

    ax1.plot(times[:-1], [Ut(t)[0] for t in times[:-1]], color="green", lw=1.5,
                label="Buffer Region")
    ax1.plot(times[:-1], [Ut(t)[1] for t in times[:-1]], color="Purple", lw=1.5,
                label="High Value Region")
    ax1.set_xlabel("Time / Infectious Period")
    ax1.set_ylabel("Proportion Treated")
    ax1.set_xlim([0, 12])
    ax1.set_ylim([-0.05, 1.1])

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([0], [0], color='r', linestyle='--'))
    labels.append("Max Expenditure")
    ax1.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=1, frameon=True,
        bbox_transform=ax4.transAxes
    )

    ax2.fill_between(
        times, 0, expense_region2, label="High Value Region", color="Purple", alpha=0.5
    )
    ax2.fill_between(
        times, expense_region2, expense_region1 + expense_region2, label="Buffer Region",
        color="green", alpha=0.5
    )
    ax2.plot(times, np.full_like(times, max_budget_rate), 'r--', label="Max Expenditure")
    ax2.set_xlabel("Time / Infectious Period")
    ax2.set_ylabel("Control Expenditure")
    ax2.set_ylim([0, max_budget_rate * 1.1])
    ax2.set_xlim([0, 12])

    ax3.plot(times[:-1], [Xt(t)[3]/N_individuals[1] for t in times[:-1]], color="Purple",
             label="High Value Region")
    ax3.plot(times[:-1], [Xt(t)[1]/N_individuals[0] for t in times[:-1]], color="green",
             label="Buffer Region")
    ax3.set_xlabel("Time / Infectious Period")
    ax3.set_ylabel("Proportion Infected")
    ax3.set_ylim([0, 0.3])

    ax4.plot(times[:-1], [(1 - (Xt(t)[3] + Xt(t)[2])/N_individuals[1]) for t in times[:-1]],
             color="Purple", label="High Value Region")
    ax4.plot(times[:-1], [(1 - (Xt(t)[1] + Xt(t)[0])/N_individuals[0]) for t in times[:-1]],
             color="green", label="Buffer Region")
    ax4.set_xlabel("Time / Infectious Period")
    ax4.set_ylabel("Proportion Removed")
    ax4.set_ylim([0, 1])

    fig.text(0.01, 0.98, '(a)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.41, 0.98, '(b)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.41, 0.56, '(c)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.67, 0.98, '(d)', transform=fig.transFigure, fontsize=9, weight="semibold")

    fig.savefig("GeneralStrategy.pdf", dpi=300, bbox_inches="tight")


def make_figure3(model):
    times = model.params['times']
    max_budget_rate = model.params['max_budget_rate']
    N_individuals = model.params['N_individuals']
    Xt, Lt, Ut, _ = model.optimised_control

    expense_region1 = np.array([Ut(t)[0]*Xt(t)[1] for t in times])
    expense_region2 = np.array([Ut(t)[1]*Xt(t)[3] for t in times])

    fig = plt.figure(figsize=(6.2, 3.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], left=0.08, right=0.65, wspace=0.4, hspace=0.5,
                           bottom=0.25, top=0.97)
    gs2 = gridspec.GridSpec(2, 1, left=0.78, right=0.96, bottom=0.25, top=0.97, hspace=0.4)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs2[1])
    ax5 = fig.add_subplot(gs2[0])

    ax1.plot(times[:-1], [Ut(t)[0] for t in times[:-1]], color="green", lw=1.5,
                label="Buffer Region")
    ax1.plot(times[:-1], [Ut(t)[1] for t in times[:-1]], color="Purple", lw=1.5,
                label="High Value Region")
    ax1.set_xlabel("Time / Infectious Period")
    ax1.set_ylabel("Proportion Treated")
    ax1.set_xlim([0, 12])
    ax1.set_ylim([-0.05, 1.1])

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([0], [0], color='r', linestyle='--'))
    labels.append("Max Expenditure")
    ax1.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.35, 0.02), ncol=3,
               frameon=True, bbox_transform=fig.transFigure)

    ax2.fill_between(
        times, 0, expense_region2, label="High Value Region", color="Purple", alpha=0.5
    )
    ax2.fill_between(times, expense_region2, expense_region1 + expense_region2,
                     label="Buffer Region", color="green", alpha=0.5)
    ax2.plot(times, np.full_like(times, max_budget_rate), 'r--', label="Max Expenditure")
    ax2.set_xlabel("Time / Infectious Period")
    ax2.set_ylabel("Control Expenditure")
    ax2.set_ylim([0, 6])
    ax2.set_xlim([0, 12])

    ax3.plot(times[:-1], [Xt(t)[3]/N_individuals[1] for t in times[:-1]], color="Purple",
             label="High Value Region")
    ax3.plot(times[:-1], [Xt(t)[1]/N_individuals[0] for t in times[:-1]], color="green",
             label="Buffer Region")
    ax3.set_xlabel("Time / Infectious Period")
    ax3.set_ylabel("Proportion Infected")
    ax3.set_ylim([0, 0.075])

    ax4.plot(times, [model.params['inf_rate'] * Xt(t)[3] for t in times],
             label='From inside value region')
    ax4.plot(times, [model.params['inf_rate'] * model.params['coupling'] * Xt(t)[1] for t in times],
             label='From outside value region')
    ax4.legend(loc='lower center', bbox_to_anchor=(0.85, 0.02), ncol=1, frameon=True,
               bbox_transform=fig.transFigure)
    ax4.set_xlabel("Time / infectious period")
    ax4.set_ylabel("Force of infection")

    ax5.plot(times[:-1], [(1 - (Xt(t)[3] + Xt(t)[2])/N_individuals[1]) for t in times[:-1]],
             color="Purple", label="High Value Region")
    ax5.plot(times[:-1], [(1 - (Xt(t)[1] + Xt(t)[0])/N_individuals[0]) for t in times[:-1]],
             color="green", label="Buffer Region")
    ax5.set_xlabel("Time / Infectious Period")
    ax5.set_ylabel("Proportion Removed")
    ax5.set_ylim([0, 1])

    fig.text(0.01, 0.98, '(a)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.41, 0.98, '(b)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.41, 0.56, '(c)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.67, 0.98, '(d)', transform=fig.transFigure, fontsize=9, weight="semibold")
    fig.text(0.67, 0.56, '(e)', transform=fig.transFigure, fontsize=9, weight="semibold")

    fig.savefig("SwitchStrategy.pdf", dpi=300, bbox_inches="tight")


def main():
    """Generate example replcias of Figure 2 and 3 from the paper."""

    # Default parameterisation
    inf_rate = 0.005
    coupling = 0.3
    ext_foi = 0
    control_rate = 0.2

    N_individuals = np.array([500, 100])
    state_init = np.array([495, 5, 100, 0])
    times = np.linspace(0, 12.5, 1001)
    control_init = np.full((2, 1001), 1, dtype=float)

    max_budget_rate = 10

    costate_final = np.array([0, 0, -1, 0])

    precision = 10

    params = {
        'inf_rate': inf_rate,
        'coupling': coupling,
        'ext_foi': ext_foi,
        'control_rate': control_rate,
        'N_individuals': N_individuals,
        'state_init': state_init,
        'times': times,
        'control_init': control_init,
        'max_budget_rate': max_budget_rate,
        'costate_final': costate_final,
        'precision': precision,
    }

    model = PatchModel(params)

    Xt, Lt, Ut, exit_text = model.run(mode="BOCOP", verbose=True)

    make_figure2(model)

    # Switch parameterisation
    inf_rate = 0.0028
    coupling = 0.3
    ext_foi = 0
    control_rate = 0.2

    N_individuals = np.array([500, 100])
    state_init = np.array([495, 5, 100, 0])
    times = np.linspace(0, 12.5, 1001)
    control_init = np.full((2, 1001), 1, dtype=float)

    max_budget_rate = 5

    costate_final = np.array([0, 0, -1, 0])

    precision = 10

    params = {
        'inf_rate': inf_rate,
        'coupling': coupling,
        'ext_foi': ext_foi,
        'control_rate': control_rate,
        'N_individuals': N_individuals,
        'state_init': state_init,
        'times': times,
        'control_init': control_init,
        'max_budget_rate': max_budget_rate,
        'costate_final': costate_final,
        'precision': precision,
    }

    model = PatchModel(params)

    Xt, Lt, Ut, exit_text = model.run(mode="BOCOP", verbose=True)

    make_figure3(model)


if __name__ == '__main__':
    mpl.rcParams.update({'font.size': 8})
    plt.style.use("seaborn-whitegrid")
    main()
