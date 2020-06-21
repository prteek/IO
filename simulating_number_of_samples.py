# author            : Prateek
# email             : prateekpatel.in@gmail.com
# description       : simulation to determine number of samples required for a survival analysis study

import os

os.system("pip install -U -r requirements.txt")

import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.special import kl_div
from scipy.stats import ranksums
import time
from scipy.interpolate import interp1d


np.random.seed(10)  # For repeatability
mean_battery_age_weeks = 5 * 4  # 5 months * 4 weeks
battery_perfectly_healthy_until_week = 2 * 4  # 2 months they are fine
fleet_size = 100  # assumption


# Generate battery life data. This should be close representation of reality and our target for modelling

fleet_age_distribution = (
    np.random.exponential(
        mean_battery_age_weeks - battery_perfectly_healthy_until_week, fleet_size
    )
    + battery_perfectly_healthy_until_week
)


plt.figure()
plt.hist(
    fleet_age_distribution, bins=np.arange(0, 49, 4), density=False, label="full fleet"
)
plt.title("Distribution of battery life for fleet")
plt.xlabel("Battery life [weeks]")
plt.ylabel("Number of fleet trucks")
plt.grid()
plt.show()


# Choose trucks and model their age

### Setting data collection experiment parameters
size_options = np.arange(0, 101, 5)[1:]  # Ignore case with 0 trucks
logging_duration_weeks = 6 * 4
no_censored = (
    False
)  # Set battery failure events to not be right-censored owing to short logging durations

# Initialising plot and result variables
auc = []
norm_auc = []
kldvg = []
plt.figure()

# Full fleet data model
observed_event = fleet_age_distribution <= 9999
time_fleet, survival_prob_fleet = kaplan_meier_estimator(
    observed_event, fleet_age_distribution
)
prob_lookup = interp1d(time_fleet, survival_prob_fleet, kind="nearest")

# np.random.seed(int(time.time()))

for n_trucks in size_options:
    n_trucks = int(n_trucks)
    # assumption is monitoring starts on perfectly new trucks
    trucks_age_weeks = np.random.choice(fleet_age_distribution, n_trucks, replace=False)

    if no_censored:
        observed_duration_weeks = 9999
    else:
        observed_duration_weeks = logging_duration_weeks

    observed_event = trucks_age_weeks <= observed_duration_weeks

    time, survival_prob = kaplan_meier_estimator(observed_event, trucks_age_weeks)

    logging_index = np.where(time <= logging_duration_weeks)

    survival_prob_fleet_matching = prob_lookup(time[logging_index])

    kldvg_i = kl_div(survival_prob_fleet_matching, survival_prob[logging_index])
    norm_auc.append(
        np.trapz(survival_prob[logging_index] * (1 - 10 * kldvg_i), time[logging_index])
    )
    auc.append(np.trapz(survival_prob[logging_index], time[logging_index]))
    kldvg.append(np.nanmean(kldvg_i))

    # Plot if number of trucks is is very low or very high (ROI)
    if n_trucks > 20 and n_trucks < 90:
        continue
    plt.step(
        time,
        survival_prob,
        where="post",
        label=str(n_trucks) + " trucks : " + str(sum(observed_event)) + " failed",
    )


plt.step(time_fleet, survival_prob_fleet, where="post", label="reality")
plt.plot(
    [logging_duration_weeks, logging_duration_weeks],
    [0, 1],
    ":k",
    label="logging_duration",
)
plt.title("Survival Probabilities")
plt.ylabel("est. probability of survival")
plt.xlabel("time [weeks]")
plt.legend()
plt.grid()
plt.show()


auc = np.array(auc)
kldvg = np.array(kldvg)
fleet_auc = np.trapz(survival_prob_fleet, time_fleet)

plt.figure()
plt.plot(size_options, auc, ":b", label="auc")
plt.plot(size_options, auc * (1 - 10 * kldvg), "b", label="KL weighted auc")
# plt.plot(size_options, kldvg*100, label="samples kldvg*100")

plt.plot(
    [0, size_options[-1]],
    [auc[-1], auc[-1]],
    ":r",
    label="auc of best model using "
    + str(logging_duration_weeks / 4)
    + " months logging",
)
plt.plot(
    [0, size_options[-1]], [auc[-1] - 1, auc[-1] - 1], ":k", label="-1 weeks best auc"
)
plt.plot(
    [0, size_options[-1]], [auc[-1] + 1, auc[-1] + 1], ":k", label="+1 weeks best auc"
)
plt.plot([0, size_options[-1]], [fleet_auc, fleet_auc], ":g", label="reality")

plt.title("AUC of survival curves")
plt.xlabel(
    "number of trucks studied over " + str(logging_duration_weeks / 4) + " months"
)
plt.ylabel("AUC [weeks]")
plt.legend()
plt.xticks(size_options[0:-1:2])
plt.grid()
plt.show()


# Close all plots to avoid hogging RAM
plt.close("all")
