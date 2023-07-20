# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from scipy import stats
from IPython.display import display
import duckdb

hv.extension("bokeh")
sns.set_theme()

sql = lambda q: duckdb.sql(q).df()

# %%

query_read_and_format_data = """
select
Title as title
, cast("World Sales (in $)" as double) as world_sales
, cast("International Sales (in $)" as double) as international_sales
, cast("Domestic Sales (in $)" as double) as domestic_sales
, strptime(coalesce(nullif("Release Date", 'NA'), 'January 01, 1900'), '%B %d, %Y') as release_date
, datepart('year', strptime(coalesce(nullif("Release Date", 'NA'), 'January 01, 1900'), '%B %d, %Y')) as release_year
, cast(regexp_extract("Movie Runtime", '(\d+) hr?', 1) as int)*60+ cast(coalesce(nullif(regexp_extract("Movie Runtime", '(\d+) min?', 1),''),0) as int) as running_time_mins
from 'docs/Highest Holywood Grossing Movies.csv'

"""

df = sql(query_read_and_format_data)

assert df["running_time_mins"].isna().sum() == 0, "Error in running time parsing"

# %%

n = 10
query_top_movies = f"""
with ranked
as
(
   select
     *
     , row_number() over(partition by release_year order by world_sales desc) as yearly_sale_rank
   from df
)
select *
from ranked
where yearly_sale_rank <= {n} and release_year >= 1990
order by release_year
"""

df_yearly_top_movies = sql(query_top_movies)


kdims = ["release_year"]
vdims = ["running_time_mins", "title"]
ds = hv.Dataset(df_yearly_top_movies, kdims=kdims, vdims=vdims)

agg = ds.aggregate("release_year", function=np.mean)
fig = (hv.Scatter(ds) * hv.Curve(agg)).opts(
    hv.opts.Curve(width=400, height=400, show_grid=True, color="k"),
    hv.opts.Scatter(size=5, tools=["hover"]),
)

display(fig)


# %%

base_window = 1995, 1999
test_year = 2018

query_test = f"""
select *
, case when release_year={test_year} then true
else false
end as release_in_test_year
, log(running_time_mins) as log_time
from df_yearly_top_movies
where (release_year >= {base_window[0]} and release_year < {base_window[1]}) or (release_year={test_year})
"""

df_test = sql(query_test)

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

g1 = sql(f"select {variable} from df_test where release_in_test_year = false")
g2 = sql(f"select {variable} from df_test where release_in_test_year = true")


def difference_of_mean(sample1, sample2):
    statistic = np.mean(sample1) - np.mean(sample2)
    return statistic


res = stats.bootstrap(
    (g1, g2), statistic=difference_of_mean, alternative="less", random_state=42
)

fig = hv.Distribution(res.bootstrap_distribution[0]) * hv.VLine(0).opts(
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
