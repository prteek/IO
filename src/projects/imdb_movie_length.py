# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from scipy import stats
from IPython.display import display

hv.extension("bokeh")
sns.set_theme()

# %%
df = (
    pd.read_csv("docs/Highest Holywood Grossing Movies.csv", index_col=0)
    .assign(
        _hr=lambda x: x["Movie Runtime"]
        .str.extract(r"(\d+) hr", flags=0, expand=True)
        .astype(float),
        _min=lambda x: x["Movie Runtime"]
        .str.extract(r"(\d+) min", flags=0, expand=True)
        .astype(float),
        running_time_mins=lambda x: x["_hr"] * 60 + x["_min"].fillna(0),
        release_date=lambda x: pd.to_datetime(x["Release Date"]),
        release_year=lambda x: x["release_date"].dt.year,
    )
    .astype(
        {
            "World Sales (in $)": float,
            "International Sales (in $)": float,
            "Domestic Sales (in $)": float,
        }
    )
    .rename(
        {
            "World Sales (in $)": "world_sales",
            "International Sales (in $)": "international_sales",
            "Domestic Sales (in $)": "domestic_sales",
        },
        axis=1,
    )
)


assert df["running_time_mins"].isna().sum() == 0, "Error in running time parsing"

# %%

n = 10
df_yearly_top_movies = (
    df.groupby(["release_year"])
    .apply(lambda x: x.nlargest(n, ["world_sales"]))
    .reset_index(drop=True)
    .query("release_year >= 1990")
)

kdims = ["release_year"]
vdims = ["running_time_mins", "Title"]
ds = hv.Dataset(df_yearly_top_movies, kdims=kdims, vdims=vdims)

agg = ds.aggregate("release_year", function=np.mean)
fig = (hv.Scatter(ds) * hv.Curve(agg)).opts(
    hv.opts.Curve(width=400, height=400, show_grid=True, color="k"),
    hv.opts.Scatter(size=5, tools=["hover"]),
)

display(overlay)


# %%

base_window = 1995, 1999
test_year = 2018
df_test = df_yearly_top_movies.query(
    "(release_year >= @base_window[0] and release_year < @base_window[1]) or (release_year==@test_year)"
).assign(
    release_in_test_year=lambda x: x["release_year"] == test_year,
    log_time=lambda x: np.log(x["running_time_mins"]),
)


ds = hv.Dataset(df_test, kdims=["release_in_test_year"])

fig = (
    ds.to(hv.Distribution, "running_time_mins")
    .overlay("release_in_test_year")
    .opts(width=400, height=400, show_grid=True)
)

display(fig)

fig = (
    ds.to(hv.Distribution, "log_time")
    .overlay("release_in_test_year")
    .opts(width=400, height=400, show_grid=True)
)

display(fig)

variable = "running_time_mins"
alpha = 0.05

g1 = df_test.query("release_in_test_year == False")[variable]
g2 = df_test.query("release_in_test_year == True")[variable]


def difference_of_mean(sample1, sample2):
    statistic = np.mean(sample1) - np.mean(sample2)
    return statistic


res = stats.bootstrap(
    (g1, g2), statistic=difference_of_mean, alternative="less", random_state=42
)

fig = hv.Distribution(res.bootstrap_distribution) * hv.VLine(0).opts(
    color="black",
    xlabel="Bootstrap difference of means",
    width=400,
    height=400,
    show_grid=True,
)
display(fig)

print(
    f"mean(difference of means) : {np.mean(res.bootstrap_distribution)}",
    "\n",
    f"proportion of samples < 0 : {np.mean(res.bootstrap_distribution < 0)}",
    "\n"
    f"Null hypothesis rejected: {(1 - np.mean(res.bootstrap_distribution < 0)) <= alpha}",
)
