# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from scipy import stats
from IPython.display import display
import duckdb
import requests
from bs4 import BeautifulSoup as BS
import imdb
import tqdm
from joblib import Parallel, delayed


hv.extension("bokeh")
sns.set_theme()

sql = lambda q: duckdb.sql(q).df()

# %%
# Scraping data

ia = imdb.Cinemagoer()

years = range(1990, 2024)
yearly_top_grossing_url = "https://www.boxofficemojo.com/year/world/{year}/"


page = requests.get(yearly_top_grossing_url.format(year=2023))
soup = BS(page.content, "html.parser")
titles = soup.find_all("td", class_="a-text-left mojo-field-type-release_group")
print(titles[0].select("a")[0].string)

# %%


def get_year_matched_movie_from_title(title: str, year: int):
    movies = ia.search_movie(title)
    for movie in movies:
        if ia.get_movie_main(movie.getID())["data"]["year"] == year:
            return movie
        else:
            continue
    return None


def get_info_from_movie(movie):
    run_time = ia.get_movie_main(movie.getID())["data"]["runtimes"][0]
    year = ia.get_movie_main(movie.getID())["data"]["year"]
    title_info = {
        "release_year": year,
        "runtime_mins": int(run_time),
    }
    return title_info


def get_info_for_title(title, year):
    movie = get_year_matched_movie_from_title(title, year)
    if movie is not None:
        title_info = get_info_from_movie(movie)
    else:
        title_info = dict()
        title_info["release_year"] = year
        title_info["runtime_mins"] = np.nan

    title_info["title"] = title
    return title_info


top_n = 10
all_titles = []
pbar = tqdm.tqdm(years, position=0)
problematic_movie_titles = [
    "300"
]  # 300 was probably released in 2006 but appears on charts in 2007 and isn't matched correctly
for year in pbar:
    pbar.set_description(str(year))
    page = requests.get(yearly_top_grossing_url.format(year=year))
    soup = BS(page.content, "html.parser")
    titles = soup.find_all("td", class_="a-text-left mojo-field-type-release_group")
    delayed_year_results = []
    for t in titles[:top_n]:
        title = t.select("a")[0].string
        if title in problematic_movie_titles:
            continue
        title_info = delayed(get_info_for_title)(title, year)
        delayed_year_results.append(title_info)

    year_results = Parallel(n_jobs=top_n, prefer="threads")(delayed_year_results)
    all_titles.extend(year_results)

df_movies = pd.DataFrame(all_titles)
df_movies.to_csv("movies_dataset.csv")

# %%
query_read_and_format_data = """
select *
from 'movies_dataset.csv'
"""

df_yearly_top_movies = sql(query_read_and_format_data)

assert df["runtime_mins"].isna().sum() == 0, "Error in running time parsing"

# %%

kdims = ["release_year"]
vdims = ["runtime_mins", "title"]
ds = hv.Dataset(df_yearly_top_movies, kdims=kdims, vdims=vdims)

agg = ds.aggregate("release_year", function=np.mean)
fig = (hv.Scatter(ds) * hv.Curve(agg)).opts(
    hv.opts.Curve(width=400, height=400, show_grid=True, color="k"),
    hv.opts.Scatter(size=5, tools=["hover"]),
)

display(fig)


# %%

base_window = 1995, 1999
test_year = 2022

# We shall try to create a tidy dataset for further exploration
query_test = f"""
select *
, case when release_year={test_year} then true
else false
end as release_in_test_year
, log(runtime_mins) as log_time
from df_yearly_top_movies
where (release_year >= {base_window[0]} and release_year < {base_window[1]}) or (release_year={test_year})
"""

df_test = sql(query_test)

ds = hv.Dataset(df_test, kdims=["release_in_test_year"])

fig = (
    ds.to(hv.Distribution, "runtime_mins")
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

variable = "runtime_mins"
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


# %%
