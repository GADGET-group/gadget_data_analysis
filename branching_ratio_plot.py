import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16

# Labels for x-axis
labels = [
    "UMADAC", "Microscopic", "MGLDM",  # theory
    "Simulated", "Constant", "Linear", "Quadratic", "Exponential", "Back-to-Back"
]

# Branching ratios
# First three = theory (no errors)
# Last six = your results
branching_ratios = np.array([
    8.51e-3,      # UMADAC
    3.98e-7,      # Microscopic
    1e-20,        # MGLDM
    9.72391e-07,  # Simulated (from excel spreadsheet)
    8.96892e-07,  # Constant
    7.98275E-07,  # Linear
    8.09096E-07,  # Quadratic
    7.95324E-07,  # Exponential
    3.44296E-08   # Back-to-back
])

# Symmetric uncertainties (set theory to 0) (currently exp is wrong)
errors = np.array([
    0, 0, 0,
    3.49389E-07,
    3.30883E-07,
    3.09936E-07,
    3.10239E-07,
    3.14536E-07,
    9.94485E-09
])

# 90% upper limits for your six models (currently wrong)
upper_limits = np.array([
    np.nan, np.nan, np.nan,   # theory
    1.90933E-06,
    5.02209E-06,
    1.74125E-06,
    1.74041E-06,
    1.78146E-06,
    2.27071E-07
])

x = np.arange(len(labels))

# ---------------------------
# FIGURE WITH BROKEN AXES
# ---------------------------

fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
    3, 1, sharex=True, figsize=(10, 8),
    gridspec_kw={'height_ratios': [1, 3, 1]}
)

# Y limits for each region
ax_top.set_ylim(1e-4, 2e-2)      # UMADAC region
# ax_mid.set_ylim(3e-8, 3e-6)      # main cluster
ax_mid.set_ylim(2e-8, 6e-6)      # main cluster
ax_bot.set_ylim(1e-21, 1e-18)    # MGLDM region

for ax in (ax_top, ax_mid, ax_bot):
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

def plot_points(ax):
    # theory
    ax.scatter(x[:3], branching_ratios[:3], marker='s', s=80)

    # results with errors
    ax.errorbar(
        x[3:], branching_ratios[3:], yerr=errors[3:],
        fmt='o', capsize=4
    )

for ax in (ax_top, ax_mid, ax_bot):
    plot_points(ax)

# ---------------------------
# SHADED UPPER LIMITS (extend upward)
# ---------------------------

for xi, ul in zip(x[3:], upper_limits[3:]):
    if not np.isnan(ul):
        # middle panel shading
        ax_mid.fill_between(
            [xi - 0.25, xi + 0.25],
            ul, ax_mid.get_ylim()[1],
            alpha=0.75, hatch='///',
            edgecolor='black', facecolor='none', linewidth=1
        )
        # extend into top panel
        ax_top.fill_between(
            [xi - 0.25, xi + 0.25],
            ax_top.get_ylim()[0], ax_top.get_ylim()[1],
            alpha=0.75, hatch='///',
            edgecolor='black', facecolor='none', linewidth=1
        )

# ---------------------------
# BREAK MARKS
# ---------------------------

d = .5  # diagonal line size
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle='none', color='k', mec='k', mew=1, clip_on=False)

ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax_mid.plot([0, 1], [1, 1], transform=ax_mid.transAxes, **kwargs)

ax_mid.plot([0, 1], [0, 0], transform=ax_mid.transAxes, **kwargs)
ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

# ---------------------------
# FORMATTING
# ---------------------------

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(labels, rotation=30, ha='right')

ax_mid.set_ylabel("Branching Ratio")
ax_bot.set_xlabel("Prediction / Model")

fig.tight_layout()
plt.savefig("branching_ratio_broken_axis.png", dpi=600)
plt.show()

# ---------------------------
# PLOTTING
# ---------------------------

x = np.arange(len(labels))

plt.figure(figsize=(10, 6))

# Plot theory points
plt.scatter(x[:3], branching_ratios[:3], marker='s', s=80, label="Theory")

# Plot your results with error bars
plt.errorbar(
    x[3:], branching_ratios[3:], yerr=errors[3:],
    fmt='o', capsize=5, label="This Work"
)

# # Plot 90% upper limit arrows
# for xi, ul in zip(x[3:], upper_limits[3:]):
#     if not np.isnan(ul):
#         plt.annotate(
#             '',
#             xy=(xi, ul),
#             xytext=(xi, ul/3),
#             arrowprops=dict(arrowstyle='-|>', lw=1.5)
#         )

# Shade excluded regions (values above upper limit)
ymax = 1e-2  # adjust to top of your plot

for xi, ul in zip(x[3:], upper_limits[3:]):
    if not np.isnan(ul):
        plt.fill_between(
            [xi - 0.25, xi + 0.25],  # width of shaded band
            ul, ymax,
            alpha=0.25,
            hatch='///',
            edgecolor='gray',
            facecolor='none',
            linewidth=0
        )
        
# Log scale for branching ratios
plt.yscale('log')

# Axis formatting
plt.xticks(x, labels, rotation=30, ha='right')
plt.ylabel("Branching Ratio")
plt.xlabel("Model / Prediction")

# Grid and layout
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save high-quality figure
# plt.savefig("branching_ratio_results.pdf")
plt.savefig("branching_ratio_results.png", dpi=600)

plt.show()
