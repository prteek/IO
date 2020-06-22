# author            : Prateek
# email             : prateekpatel.in@gmail.com
# description       : simulation to determine number of samples required for a survival analysis study

import os

# os.system("pip install -U -r requirements.txt")

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.special import kl_div
import time as TT
from scipy.interpolate import interp1d
import streamlit as st

st.title('Estimating number of samples for Survival Analysis study')

np.random.seed(9)  # For repeatability

"""### Assumptions: """

fleet_size = st.slider("Fleet size (# of trucks)", 50, 500, 100, step=50)  # assumption


"""
#### 1 month = 4 weeks 
"""

battery_perfectly_healthy_until_month = st.slider("No batteries fail until [months]", 0, 12, 2, step=1)   # 2 months they are fine
battery_perfectly_healthy_until_week  = battery_perfectly_healthy_until_month*4

mean_battery_age_months = st.slider('Mean battery life [months] ', battery_perfectly_healthy_until_month, 12, max(5, battery_perfectly_healthy_until_month))  # 5 months * 4 weeks
mean_battery_age_weeks = mean_battery_age_months*4



# Generate battery life data. This should be close representation of reality and our target for modelling

fleet_age_distribution = (
    np.random.exponential(
        mean_battery_age_weeks - battery_perfectly_healthy_until_week, fleet_size
    )
    + battery_perfectly_healthy_until_week
)

"""
### This is the first source of uncertainity. The lifetimes of batteries in the fleet has a wide distribution.
### This is assumed to be ```exponential``` as understood from failure patterns of machine components.

"""

plt.figure()
plt.hist(
    fleet_age_distribution, bins=np.arange(0, 49, 4), density=False, label="full fleet"
)
plt.title("Distribution of battery life for fleet")
plt.xlabel("Battery life [weeks]")
plt.ylabel("Number of fleet trucks")
plt.grid()
plt.show()
st.pyplot()

# Choose trucks and model their age

### Setting data collection experiment parameters
size_options = np.arange(0, fleet_size+1, 5)[1:]  # Ignore case with 0 trucks
no_censored = (
    False
)  # Set battery failure events to not be right-censored owing to short logging durations


"""### Studying Battery failures"""

logging_duration_months = st.slider('Logging duration', 0,12,6,step=1)
logging_duration_weeks  = logging_duration_months*4

"""
During the logging duration (%d months) trucks are chosen and each one of their batteries may or may-not survive past this logging duration.

Ideally the more failures are captured during logging, the better.

The logging duration thus becomes important in modelling accuracy and its effect can be seen later in AUC (area under the curve) plot also.

""" % (logging_duration_weeks/4)


# Initialising plot and result variables
auc = []
norm_auc = []
kldvg = []
plt.figure()
kmf = KaplanMeierFitter()
# Full fleet data model
observed_event = fleet_age_distribution <= 9999
time_fleet = np.arange(int(max(fleet_age_distribution))+1)
kmf.fit(
    fleet_age_distribution, observed_event, timeline=time_fleet)
survival_prob_fleet = np.array(kmf.survival_function_.KM_estimate)
prob_lookup = interp1d(time_fleet, survival_prob_fleet, kind='nearest')


button_input =  st.button('Randomise and re-run')

if button_input:
	np.random.seed(int(TT.time()))

for n_trucks in size_options:
    n_trucks = int(n_trucks)
    # assumption is monitoring starts on perfectly new trucks
    trucks_age_weeks = np.random.choice(fleet_age_distribution, n_trucks, replace=False)

    if no_censored:
        observed_duration_weeks = 9999
    else:
        observed_duration_weeks = logging_duration_weeks

    kmf = KaplanMeierFitter()
    observed_event = trucks_age_weeks <= observed_duration_weeks

    time = np.arange(logging_duration_weeks+1)
    kmf.fit(trucks_age_weeks, observed_event, timeline=time)
    survival_prob = np.array(kmf.survival_function_.KM_estimate)

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
st.pyplot()

auc = np.array(auc)
norm_auc = np.array(norm_auc)
kldvg = np.array(kldvg)
fleet_auc = np.trapz(survival_prob_fleet, time_fleet)


"""
#### The plot below represents that for various selection of number of trucks to be studied, how close the modelled behaviour can be to:
* Best model that can be possible with %d months of logging
* Reality i.e. if entire fleet were studied until all batteries failed

AUC (Area under the Survival Curve) of a model can be considered an average life prediction (weeks) by that model.

The objective here is to understand that random influences can cause a good AUC (close to best model) but really a stable point is where the variations get less and predictions are more stable. 

This may be considered analogous to convergence behaviour of PID.

#### So our target number of samples (ideally) should be:
* A number about which AUC variations are less
* The model average life prediction (AUC) is within +/- 1 week of best model's AUC


Also, notice that the logging duration brings *best model* closer to *reality*.
"""%(logging_duration_weeks/4)


plt.figure()
plt.plot(size_options, auc//1, ":b", label="auc")
plt.plot(size_options, norm_auc//1, "b", label="KL weighted auc")
# plt.plot(size_options, kldvg*100, label="samples kldvg*100")

plt.plot(
    [0, size_options[-1]],
    [auc[-1]//1, auc[-1]//1],
    ":r",
    label="auc of best model using "
    + str(logging_duration_weeks / 4)
    + " months logging",
)
plt.plot(
    [0, size_options[-1]], [auc[-1]//1 - 1, auc[-1]//1 - 1], ":k", label="-1 weeks best auc"
)
plt.plot(
    [0, size_options[-1]], [auc[-1]//1 + 1, auc[-1]//1 + 1], ":k", label="+1 weeks best auc"
)
plt.plot([0, size_options[-1]], [fleet_auc//1, fleet_auc//1], ":g", label="reality")

plt.title("AUC (area under curve) of survival curves")
plt.xlabel(
    "number of trucks studied over " + str(logging_duration_weeks / 4) + " months"
)
plt.ylabel("AUC [weeks]")
plt.legend()
plt.xticks(size_options[0:-1:2])
plt.grid()
plt.show()
st.pyplot()

# Close all plots to avoid hogging RAM
plt.close("all")
