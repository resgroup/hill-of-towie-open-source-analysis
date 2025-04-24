# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# if you don't have the following libraries already installed, uncomment the line below and run it
# # !pip install pandas plotly

# %%
import pandas as pd
import plotly.express as px
from IPython.display import display

# setting the location of the source data
from hot_open.paths import get_analyses_directory

analysis_dir = get_analyses_directory(analysis_name="hill-of-towie-open-source-analysis")
train_data_fpath = analysis_dir / "wedowind_competition_input_data" / "train_dataset.parquet"
submission_data_fpath = analysis_dir / "wedowind_competition_input_data" / "submission_dataset.parquet"


# %%
class Cols:
    """Class to hold the column names of the dataset. Which allows for tab-completion and minimizes typos."""

    TIMESTAMP = "TimeStamp"
    TURBINE_ID = "turbine_id"
    WINDSPEED_MEAN = "wtc_AcWindSp_mean"
    OP_TIMEON = "wtc_ScReToOp_timeon"
    ACTIVEPOWER_MEAN = "wtc_ActPower_mean"


# %% [markdown]
# ## Load the date into memory

# %%
input_df = pd.read_parquet(train_data_fpath)
input_df.head(3)

# %% [markdown]
# ## Exploring the data with interactive plots

# %%
px.scatter(
    (
        # grabbing some random subset of the data
        input_df.sample(n=10_000)
        # making turbine id categorical to be able to select turbine from the legend
        .astype({Cols.TURBINE_ID: "category"})
    ),
    x=Cols.WINDSPEED_MEAN,
    y=Cols.ACTIVEPOWER_MEAN,
    color=Cols.TURBINE_ID,
)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# Splitting the training data into a training subset (70%) and a test subset (30%)
# so we can check our model performance on unseen data (the test set).
#
# We'll also pivot the data into a "wide" format, where each column is repeated for each turbine. So instead of:
#
# | turbine_id | TimeStamp | colA | colB |
# | --- | --- | --- | --- |
# | 1 | 2000-01-01 00:00 | 1 | 5 |
# | 2 | 2000-01-01 00:00 | 2 | 6 |
# | 1 | 2000-01-01 00:10 | 3 | 7 |
# | 2 | 2000-01-01 00:10 | 4 | 8 |
#
# We'll have:
#
# | TimeStamp | colA;1 | colA;2 | colB;1 | colB;2 |
# | --- | --- | --- | --- | --- |
# | 2000-01-01 00:00 | 1 | 2 | 5 | 6 |
# | 2000-01-01 00:10 | 3 | 4 | 7 | 8 |
#
# And finally we'll separate "target" column (active power of turbine 1)
# from the "feature" columns (all columns of the other turbines).

# %%
target_turbine = 1
target_field = Cols.ACTIVEPOWER_MEAN
test_fraction = 0.3

ts_min, ts_max = input_df[Cols.TIMESTAMP].min(), input_df[Cols.TIMESTAMP].max()
test_subset_start = ts_min + (ts_max - ts_min) * (1 - test_fraction)
is_train_subset = input_df[Cols.TIMESTAMP] < test_subset_start
FULLY_ACTIVE_VALUE = 600  # fully operating during 10min interval


def _make_wide(d: pd.DataFrame) -> pd.DataFrame:
    d = d.set_index([Cols.TURBINE_ID, Cols.TIMESTAMP]).unstack(0)  # type: ignore[assignment]
    d.columns = [f"{col};{turbine_id}" for col, turbine_id in d.columns]  # type: ignore[misc,assignment,has-type]
    return d


def _split_into_X_y(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = f"{target_field};{target_turbine}"
    feature_cols = [i for i in d.columns if not i.endswith(f";{target_turbine}")]
    target_not_fully_operational = d[f"{Cols.OP_TIMEON};{target_turbine}"] < FULLY_ACTIVE_VALUE

    X = d[feature_cols]
    # if non-fully operational target turbine data, set as NaN
    y = d[target_col].mask(target_not_fully_operational)
    return X, y


X_train, y_train = input_df[is_train_subset].pipe(_make_wide).pipe(_split_into_X_y)
X_test, y_test = input_df[~is_train_subset].pipe(_make_wide).pipe(_split_into_X_y)

# %%
display(X_train.head(3))
display(y_train.head(3))

# %%
display(X_test.head(3))
display(y_test.head(3))


# %% [markdown]
# ## Define the model
#
# We'll use a simple model that just takes the estimates the Active Power of the target
# turbine as the average of the Active Power of all other fully-operating turbines.


# %%
# we could write it as a simple function...
def mean_active_power_of_fully_operating_non_target_turbines(X: pd.DataFrame) -> pd.Series:
    """Calculate mean of Active Power for all other turbines (non-target) that are fully operating."""
    others_active_power = X.filter(regex=rf"^{Cols.ACTIVEPOWER_MEAN}")
    others_is_fully_operating = X.filter(regex=rf"^{Cols.OP_TIMEON}") == FULLY_ACTIVE_VALUE
    return others_active_power.mask(~others_is_fully_operating.to_numpy()).mean(axis=1)


# but better yet, we can make it a class in the style of the scikit-learn library
class SimpleModel:
    """Simple model that takes Active Power from all other fully operating wind turbines and averages them."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleModel":
        # this simple model does not need training (coefficient fitting)
        # but this is where you can put your logic to train your model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return mean_active_power_of_fully_operating_non_target_turbines(X)


# %% [markdown]
# ## Evaluate the model

# %%
simple_model = SimpleModel()
simple_model.fit(X=X_train, y=y_train)
y_pred = simple_model.predict(X_test)

# %%
root_mean_squared_error = (y_pred - y_test).pow(2).mean() ** 0.5
print(f"The RMSE for model is {root_mean_squared_error:.2f}")

# %% [markdown]
# ### visualising the performance

# %%
px.scatter(
    pd.DataFrame({"Actual": y_test, "Predicted": y_pred}),
    x="Actual",
    y="Predicted",
    opacity=0.2,
    title=f"Turbine {target_turbine} {target_field}",
)

# %% [markdown]
# ## Predicting on the Submission Data

# %%
# load and reshape data
X_submission = pd.read_parquet(submission_data_fpath).pipe(_make_wide)
X_submission.head(3)

# %%
# use the model to predict target turbine active power
submission_predictions = simple_model.predict(X_submission).rename("prediciton")

# %% [markdown]
# ### Exporting the prediciton for submission
# the file will be store in the same directory as this notebook

# %%
submission_predictions.to_csv("sample_model_submission.csv")
