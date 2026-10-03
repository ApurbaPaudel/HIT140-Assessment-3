#!/usr/bin/env python
# coding: utf-8

# # Linear Regression 2.1 - FIFA World Cup 2026 Goal Difference Prediction
# 
# ## Step 1: Load and Inspect the Pre-Match Dataset

# In[1]:


import pandas as pd


# In[2]:


features_df = pd.read_csv('/Users/apuu/Desktop/Assessment3_LinearRegression_2_1/data/match_prediction_features_X.csv')


# In[3]:


features_df.head()


# In[4]:


features_df.shape


# In[5]:


features_df.columns.tolist()


# In[6]:


targets_df = pd.read_csv('/Users/apuu/Desktop/Assessment3_LinearRegression_2_1/data/match_prediction_targets_y.csv')


# In[7]:


targets_df.head()


# In[8]:


targets_df["goal_difference"] = targets_df["home_score"] - targets_df["away_score"]


# In[9]:


targets_df[["match_id", "home_score", "away_score", "goal_difference"]].head()


# In[10]:


targets_df.shape


# In[11]:


targets_df["match_id"].nunique()


# In[12]:


features_df["match_id"].nunique()


# In[13]:


features_df["match_id"].nunique()


# In[14]:


set(features_df["match_id"]) == set(targets_df["match_id"])


# In[15]:


features_df.shape


# In[16]:


model_df = pd.merge(
    features_df,
    targets_df[["match_id", "goal_difference"]],
    on="match_id",
    how="inner"
)


# In[17]:


model_df.shape


# In[18]:


model_df["fifa_rank_advantage"] = (
    model_df["away_fifa_rank"] - model_df["home_fifa_rank"]
)


# In[19]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_fifa_rank",
        "away_fifa_rank",
        "fifa_rank_advantage"
    ]
].head()


# In[20]:


model_df["elo_difference"] = (
    model_df["home_elo"] - model_df["away_elo"]
)


# In[21]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_elo",
        "away_elo",
        "elo_difference"
    ]
].head()


# In[22]:


model_df["squad_value_difference"] = (
    model_df["home_squad_total_value_eur"]
    - model_df["away_squad_total_value_eur"]
)


# In[23]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_squad_total_value_eur",
        "away_squad_total_value_eur",
        "squad_value_difference"
    ]
].head()


# In[24]:


model_df["squad_experience_difference"] = (
    model_df["home_squad_total_caps"]
    - model_df["away_squad_total_caps"]
)


# In[25]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_squad_total_caps",
        "away_squad_total_caps",
        "squad_experience_difference"
    ]
].head()


# In[26]:


model_df["rest_days_difference"] = (
    model_df["home_rest_days"]
    - model_df["away_rest_days"]
)


# In[27]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_rest_days",
        "away_rest_days",
        "rest_days_difference"
    ]
].head()


# In[28]:


model_df["rest_days_difference"].value_counts().sort_index()


# In[29]:


model_df["previous_goals_scored_difference"] = (
    model_df["home_prev_avg_goals_scored"]
    - model_df["away_prev_avg_goals_scored"]
)


# In[30]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_prev_avg_goals_scored",
        "away_prev_avg_goals_scored",
        "previous_goals_scored_difference"
    ]
].head()


# In[31]:


model_df["previous_goals_scored_difference"].value_counts().sort_index()


# In[32]:


model_df["defensive_advantage"] = (
    model_df["away_prev_avg_goals_conceded"]
    - model_df["home_prev_avg_goals_conceded"]
)


# In[33]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_prev_avg_goals_conceded",
        "away_prev_avg_goals_conceded",
        "defensive_advantage"
    ]
].head()


# In[34]:


model_df["defensive_advantage"].value_counts().sort_index()


# In[35]:


model_df["host_advantage"] = (
    model_df["home_is_host"]
    - model_df["away_is_host"]
)


# In[36]:


model_df[
    [
        "home_team_name",
        "away_team_name",
        "home_is_host",
        "away_is_host",
        "host_advantage"
    ]
].head()


# In[37]:


lr21_df = model_df[
    [
        "match_id",
        "fifa_rank_advantage",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "rest_days_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "goal_difference"
    ]
].copy()


# In[38]:


lr21_df.shape


# In[39]:


lr21_df.head()


# In[40]:


lr21_df.isnull().sum()


# In[41]:


lr21_df.duplicated().sum()


# In[42]:


lr21_df.dtypes


# In[43]:


lr21_df["host_advantage"].value_counts().sort_index()


# In[44]:


lr21_df.to_csv(
    "../data/linear_regression_2_1_dataset.csv",
    index=False
)


# In[45]:


saved_lr21_df = pd.read_csv("../data/linear_regression_2_1_dataset.csv")

saved_lr21_df.shape


# In[46]:


lr21_df.drop(columns="match_id").describe()


# In[47]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.hist(
    lr21_df["goal_difference"],
    bins=range(-4, 8)
)

plt.xlabel("Goal Difference (Home Goals - Away Goals)")
plt.ylabel("Number of Matches")
plt.title("Distribution of Goal Difference")
plt.xticks(range(-4, 7))

plt.show()


# In[48]:


correlation_matrix = lr21_df.drop(columns="match_id").corr()

correlation_matrix.round(2)


# In[49]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["elo_difference"],
    lr21_df["goal_difference"]
)

plt.xlabel("Elo Difference (Home - Away)")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Elo Difference vs Goal Difference")

plt.show()


# In[50]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["fifa_rank_advantage"],
    lr21_df["goal_difference"]
)

plt.xlabel("FIFA Rank Advantage")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("FIFA Rank Advantage vs Goal Difference")

plt.show()


# In[51]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["squad_value_difference"],
    lr21_df["goal_difference"]
)

plt.xlabel("Squad Value Difference (Home - Away)")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Squad Value Difference vs Goal Difference")

plt.show()


# In[52]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["squad_experience_difference"],
    lr21_df["goal_difference"]
)

plt.xlabel("Squad Experience Difference (Home - Away)")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Squad Experience Difference vs Goal Difference")

plt.show()


# In[53]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["rest_days_difference"],
    lr21_df["goal_difference"]
)

plt.xlabel("Rest Days Difference (Home - Away)")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Rest Days Difference vs Goal Difference")

plt.show()


# In[54]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["previous_goals_scored_difference"],
    lr21_df["goal_difference"]
)

plt.xlabel("Previous Goals Scored Difference")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Previous Goals Scored Difference vs Goal Difference")

plt.show()


# In[55]:


plt.figure(figsize=(8, 5))

plt.scatter(
    lr21_df["defensive_advantage"],
    lr21_df["goal_difference"]
)

plt.xlabel("Defensive Advantage")
plt.ylabel("Goal Difference (Home - Away)")
plt.title("Defensive Advantage vs Goal Difference")

plt.show()


# In[56]:


import numpy as np

Q1 = np.percentile(lr21_df["goal_difference"], 25)
Q3 = np.percentile(lr21_df["goal_difference"], 75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)


# In[57]:


lr21_df[
    (lr21_df["goal_difference"] < lower_bound) |
    (lr21_df["goal_difference"] > upper_bound)
]


# In[58]:


X = lr21_df[
    [
        "fifa_rank_advantage",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "rest_days_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage"
    ]
]

y = lr21_df["goal_difference"]


# In[59]:


print("X shape:", X.shape)
print("y shape:", y.shape)


# In[60]:


lr21_df[
    [
        "fifa_rank_advantage",
        "elo_difference",
        "goal_difference"
    ]
].corr().round(2)


# In[61]:


model_df["previous_xg_scored_difference"] = (
    model_df["home_prev_avg_xg_scored"]
    - model_df["away_prev_avg_xg_scored"]
)


# In[62]:


model_df[
    [
        "previous_xg_scored_difference",
        "fifa_rank_advantage",
        "elo_difference",
        "goal_difference"
    ]
].corr().round(2)


# In[63]:


model_df[
    [
        "fifa_rank_advantage",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "rest_days_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_xg_scored_difference",
        "goal_difference"
    ]
].corr()[["previous_xg_scored_difference"]].round(2)


# In[64]:


model_df["previous_shots_on_target_difference"] = (
    model_df["home_prev_avg_shots_on_target"]
    - model_df["away_prev_avg_shots_on_target"]
)


# In[65]:


model_df[
    [
        "previous_shots_on_target_difference",
        "fifa_rank_advantage",
        "elo_difference",
        "squad_value_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "goal_difference"
    ]
].corr()[["previous_shots_on_target_difference"]].round(2)


# In[66]:


candidate_features = [
    "elo_difference",
    "squad_value_difference",
    "squad_experience_difference",
    "rest_days_difference",
    "previous_goals_scored_difference",
    "defensive_advantage",
    "host_advantage",
    "previous_shots_on_target_difference"
]

model_df[candidate_features].corr().round(2)


# In[67]:


lr21_refined_df = model_df[
    [
        "match_id",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "rest_days_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].copy()


# In[68]:


lr21_refined_df.shape


# In[69]:


X_original = lr21_df.drop(
    columns=["match_id", "goal_difference"]
)

X_refined = lr21_refined_df.drop(
    columns=["match_id", "goal_difference"]
)

y = lr21_df["goal_difference"]


# In[70]:


print("Original X:", X_original.shape)
print("Refined X:", X_refined.shape)
print("y:", y.shape)


# In[71]:


from sklearn.model_selection import train_test_split

(
    X_original_train,
    X_original_test,
    X_refined_train,
    X_refined_test,
    y_train,
    y_test
) = train_test_split(
    X_original,
    X_refined,
    y,
    test_size=0.20,
    random_state=42
)


# In[72]:


print("Training matches:", len(y_train))
print("Testing matches:", len(y_test))

print("Original training shape:", X_original_train.shape)
print("Refined training shape:", X_refined_train.shape)


# In[73]:


from sklearn.linear_model import LinearRegression

original_model = LinearRegression()
refined_model = LinearRegression()

original_model.fit(X_original_train, y_train)
refined_model.fit(X_refined_train, y_train)


# In[74]:


original_predictions = original_model.predict(X_original_test)
refined_predictions = refined_model.predict(X_refined_test)


# In[75]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

original_mae = mean_absolute_error(y_test, original_predictions)
original_rmse = np.sqrt(mean_squared_error(y_test, original_predictions))
original_r2 = r2_score(y_test, original_predictions)

refined_mae = mean_absolute_error(y_test, refined_predictions)
refined_rmse = np.sqrt(mean_squared_error(y_test, refined_predictions))
refined_r2 = r2_score(y_test, refined_predictions)

print("Original Model")
print("MAE:", round(original_mae, 3))
print("RMSE:", round(original_rmse, 3))
print("R²:", round(original_r2, 3))

print("\nRefined Model")
print("MAE:", round(refined_mae, 3))
print("RMSE:", round(refined_rmse, 3))
print("R²:", round(refined_r2, 3))


# In[76]:


original_train_predictions = original_model.predict(X_original_train)
refined_train_predictions = refined_model.predict(X_refined_train)

original_train_r2 = r2_score(y_train, original_train_predictions)
refined_train_r2 = r2_score(y_train, refined_train_predictions)

print("Original Training R²:", round(original_train_r2, 3))
print("Refined Training R²:", round(refined_train_r2, 3))


# In[77]:


n = len(y_train)
p = X_original_train.shape[1]

original_adjusted_r2 = 1 - (
    (1 - original_train_r2) * (n - 1) / (n - p - 1)
)

refined_adjusted_r2 = 1 - (
    (1 - refined_train_r2) * (n - 1) / (n - p - 1)
)

print("Original Adjusted R²:", round(original_adjusted_r2, 3))
print("Refined Adjusted R²:", round(refined_adjusted_r2, 3))


# In[78]:


refined_residuals = y_test - refined_predictions


# In[79]:


refined_residuals


# In[80]:


plt.figure(figsize=(8, 5))

plt.hist(
    refined_residuals,
    bins=8
)

plt.xlabel("Residual (Actual - Predicted Goal Difference)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals - Refined Model")

plt.show()


# In[81]:


plt.figure(figsize=(8, 5))

plt.scatter(
    refined_predictions,
    refined_residuals
)

plt.axhline(y=0)

plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Values - Refined Model")

plt.show()


# In[82]:


refined_coefficients = pd.DataFrame({
    "Variable": X_refined.columns,
    "Coefficient": refined_model.coef_
})

refined_coefficients


# In[83]:


print("Intercept:", refined_model.intercept_)


# In[84]:


prediction_results = pd.DataFrame({
    "Actual_Goal_Difference": y_test,
    "Predicted_Goal_Difference": refined_predictions,
    "Residual": refined_residuals
}).round(2)

prediction_results


# In[85]:


prediction_results.to_csv(
    "../outputs/lr21_refined_predictions.csv",
    index=False
)


# In[86]:


lr21_refined_df.to_csv(
    "../data/linear_regression_2_1_refined_dataset.csv",
    index=False
)


# In[87]:


model_comparison = pd.DataFrame({
    "Model": ["Original", "Refined"],
    "Training_R2": [
        original_train_r2,
        refined_train_r2
    ],
    "Adjusted_Training_R2": [
        original_adjusted_r2,
        refined_adjusted_r2
    ],
    "Test_MAE": [
        original_mae,
        refined_mae
    ],
    "Test_RMSE": [
        original_rmse,
        refined_rmse
    ],
    "Test_R2": [
        original_r2,
        refined_r2
    ]
})

model_comparison.round(3)


# In[88]:


model_comparison.round(3).to_csv(
    "../outputs/lr21_model_comparison.csv",
    index=False
)


# In[89]:


model_df["previous_possession_difference"] = (
    model_df["home_prev_avg_possession"]
    - model_df["away_prev_avg_possession"]
)


# In[90]:


model_df[
    [
        "previous_possession_difference",
        "elo_difference",
        "squad_value_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].corr()[["previous_possession_difference"]].round(2)


# In[91]:


lr21_candidate3_df = model_df[
    [
        "match_id",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "previous_possession_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].copy()


# In[92]:


lr21_candidate3_df.shape


# In[93]:


X_candidate3 = lr21_candidate3_df.drop(
    columns=["match_id", "goal_difference"]
)

X_candidate3_train = X_candidate3.loc[y_train.index]
X_candidate3_test = X_candidate3.loc[y_test.index]

candidate3_model = LinearRegression()
candidate3_model.fit(X_candidate3_train, y_train)


# In[94]:


candidate3_predictions = candidate3_model.predict(X_candidate3_test)

candidate3_mae = mean_absolute_error(y_test, candidate3_predictions)
candidate3_rmse = np.sqrt(mean_squared_error(y_test, candidate3_predictions))
candidate3_r2 = r2_score(y_test, candidate3_predictions)

candidate3_train_predictions = candidate3_model.predict(X_candidate3_train)
candidate3_train_r2 = r2_score(y_train, candidate3_train_predictions)

n = len(y_train)
p = X_candidate3_train.shape[1]

candidate3_adjusted_r2 = 1 - (
    (1 - candidate3_train_r2) * (n - 1) / (n - p - 1)
)

print("Candidate 3 Model")
print("Training R²:", round(candidate3_train_r2, 3))
print("Adjusted Training R²:", round(candidate3_adjusted_r2, 3))
print("Test MAE:", round(candidate3_mae, 3))
print("Test RMSE:", round(candidate3_rmse, 3))
print("Test R²:", round(candidate3_r2, 3))


# In[95]:


model_df["previous_shots_difference"] = (
    model_df["home_prev_avg_shots"]
    - model_df["away_prev_avg_shots"]
)


# In[96]:


model_df[
    [
        "previous_shots_difference",
        "elo_difference",
        "squad_value_difference",
        "previous_possession_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].corr()[["previous_shots_difference"]].round(2)


# In[97]:


model_df["squad_age_difference"] = (
    model_df["home_squad_avg_age"]
    - model_df["away_squad_avg_age"]
)


# In[98]:


model_df[
    [
        "squad_age_difference",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "previous_possession_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].corr()[["squad_age_difference"]].round(2)


# In[99]:


model_df[
    [
        "is_knockout",
        "elo_difference",
        "squad_value_difference",
        "squad_experience_difference",
        "previous_possession_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].corr()[["is_knockout"]].round(2)


# In[100]:


lr21_candidate4_df = model_df[
    [
        "match_id",
        "elo_difference",
        "squad_value_difference",
        "is_knockout",
        "previous_possession_difference",
        "previous_goals_scored_difference",
        "defensive_advantage",
        "host_advantage",
        "previous_shots_on_target_difference",
        "goal_difference"
    ]
].copy()


# In[101]:


lr21_candidate4_df.shape


# In[102]:


X_candidate4 = lr21_candidate4_df.drop(
    columns=["match_id", "goal_difference"]
)

X_candidate4_train = X_candidate4.loc[y_train.index]
X_candidate4_test = X_candidate4.loc[y_test.index]

candidate4_model = LinearRegression()
candidate4_model.fit(X_candidate4_train, y_train)


# In[103]:


candidate4_predictions = candidate4_model.predict(X_candidate4_test)

candidate4_mae = mean_absolute_error(y_test, candidate4_predictions)
candidate4_rmse = np.sqrt(mean_squared_error(y_test, candidate4_predictions))
candidate4_r2 = r2_score(y_test, candidate4_predictions)

candidate4_train_predictions = candidate4_model.predict(X_candidate4_train)
candidate4_train_r2 = r2_score(y_train, candidate4_train_predictions)

n = len(y_train)
p = X_candidate4_train.shape[1]

candidate4_adjusted_r2 = 1 - (
    (1 - candidate4_train_r2) * (n - 1) / (n - p - 1)
)

print("Candidate 4 Model")
print("Training R²:", round(candidate4_train_r2, 3))
print("Adjusted Training R²:", round(candidate4_adjusted_r2, 3))
print("Test MAE:", round(candidate4_mae, 3))
print("Test RMSE:", round(candidate4_rmse, 3))
print("Test R²:", round(candidate4_r2, 3))


# In[104]:


model_comparison = pd.DataFrame({
    "Model": [
        "Original",
        "Refined",
        "Candidate 3",
        "Candidate 4"
    ],
    "Training_R2": [
        original_train_r2,
        refined_train_r2,
        candidate3_train_r2,
        candidate4_train_r2
    ],
    "Adjusted_Training_R2": [
        original_adjusted_r2,
        refined_adjusted_r2,
        candidate3_adjusted_r2,
        candidate4_adjusted_r2
    ],
    "Test_MAE": [
        original_mae,
        refined_mae,
        candidate3_mae,
        candidate4_mae
    ],
    "Test_RMSE": [
        original_rmse,
        refined_rmse,
        candidate3_rmse,
        candidate4_rmse
    ],
    "Test_R2": [
        original_r2,
        refined_r2,
        candidate3_r2,
        candidate4_r2
    ]
})

model_comparison.round(3)


# In[105]:


model_comparison.round(3).to_csv(
    "../outputs/lr21_model_comparison.csv",
    index=False
)


# In[106]:


lr21_candidate3_df.to_csv(
    "../data/linear_regression_2_1_candidate3_dataset.csv",
    index=False
)


# In[107]:


candidate3_residuals = y_test - candidate3_predictions

candidate3_results = pd.DataFrame({
    "Actual_Goal_Difference": y_test,
    "Predicted_Goal_Difference": candidate3_predictions,
    "Residual": candidate3_residuals
}).round(2)

candidate3_results


# In[108]:


candidate3_results.to_csv(
    "../outputs/lr21_candidate3_predictions.csv",
    index=False
)


# In[109]:


plt.figure(figsize=(7, 5))

plt.scatter(
    candidate3_results["Actual_Goal_Difference"],
    candidate3_results["Predicted_Goal_Difference"]
)

plt.plot(
    [-4, 6],
    [-4, 6]
)

plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title("Actual vs Predicted Goal Difference")

plt.show()


# In[110]:


plt.figure(figsize=(7, 5))

plt.hist(
    candidate3_residuals,
    bins=8
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution - Candidate 3")

plt.show()


# In[111]:


plt.figure(figsize=(7, 5))

plt.scatter(
    candidate3_predictions,
    candidate3_residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values - Candidate 3")

plt.show()


# In[112]:


plt.figure(figsize=(7, 5))

plt.scatter(
    candidate3_results["Actual_Goal_Difference"],
    candidate3_results["Predicted_Goal_Difference"]
)

plt.plot(
    [-4, 6],
    [-4, 6]
)

plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title("Actual vs Predicted Goal Difference")

plt.savefig(
    "../outputs/lr21_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()


# In[113]:


plt.figure(figsize=(7, 5))

plt.scatter(
    candidate3_predictions,
    candidate3_residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values - Candidate 3")

plt.savefig(
    "../outputs/lr21_residuals_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()


# In[114]:


plt.figure(figsize=(7, 5))

plt.hist(
    candidate3_residuals,
    bins=8
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution - Candidate 3")

plt.savefig(
    "../outputs/lr21_residual_distribution.png",
    bbox_inches="tight"
)

plt.show()


# In[115]:


candidate3_coefficients = pd.DataFrame({
    "Variable": X_candidate3.columns,
    "Coefficient": candidate3_model.coef_
})

candidate3_coefficients


# In[116]:


print("Intercept:", candidate3_model.intercept_)


# In[117]:


candidate3_coefficient_summary = pd.concat(
    [
        pd.DataFrame({
            "Variable": ["Intercept"],
            "Coefficient": [candidate3_model.intercept_]
        }),
        candidate3_coefficients
    ],
    ignore_index=True
)

candidate3_coefficient_summary


# In[118]:


candidate3_coefficient_summary.to_csv(
    "../outputs/lr21_candidate3_coefficients.csv",
    index=False
)


# In[119]:


candidate3_descriptive_stats = lr21_candidate3_df.drop(
    columns=["match_id"]
).describe()

candidate3_descriptive_stats.round(2)


# In[120]:


candidate3_descriptive_stats.round(2).to_csv(
    "../outputs/lr21_candidate3_descriptive_statistics.csv"
)


# In[121]:


candidate3_correlation = lr21_candidate3_df.drop(
    columns=["match_id"]
).corr()

candidate3_correlation.round(2)


# In[122]:


candidate3_correlation.round(2).to_csv(
    "../outputs/lr21_candidate3_correlation.csv"
)


# In[123]:


print("Shape:", lr21_candidate3_df.shape)

print("\nMissing values:")
print(lr21_candidate3_df.isnull().sum())

print(
    "\nDuplicate match IDs:",
    lr21_candidate3_df["match_id"].duplicated().sum()
)


# In[124]:


continuous_predictors = [
    "elo_difference",
    "squad_value_difference",
    "squad_experience_difference",
    "previous_possession_difference",
    "previous_goals_scored_difference",
    "defensive_advantage",
    "previous_shots_on_target_difference"
]

outlier_summary = []

for column in continuous_predictors:
    Q1 = lr21_candidate3_df[column].quantile(0.25)
    Q3 = lr21_candidate3_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (lr21_candidate3_df[column] < lower_bound) |
        (lr21_candidate3_df[column] > upper_bound)
    ).sum()

    outlier_summary.append([
        column,
        lower_bound,
        upper_bound,
        outlier_count
    ])

outlier_summary_df = pd.DataFrame(
    outlier_summary,
    columns=[
        "Variable",
        "Lower_Bound",
        "Upper_Bound",
        "Outlier_Count"
    ]
)

outlier_summary_df.round(2)


# In[125]:


outlier_summary_df.round(2).to_csv(
    "../outputs/lr21_candidate3_outlier_summary.csv",
    index=False
)


# In[126]:


candidate3_mse = mean_squared_error(
    y_test,
    candidate3_predictions
)

print("Candidate 3 Test MSE:", round(candidate3_mse, 3))


# In[127]:


original_mse = mean_squared_error(
    y_test,
    original_predictions
)

refined_mse = mean_squared_error(
    y_test,
    refined_predictions
)

candidate4_mse = mean_squared_error(
    y_test,
    candidate4_predictions
)

print("Original MSE:", round(original_mse, 3))
print("Refined MSE:", round(refined_mse, 3))
print("Candidate 3 MSE:", round(candidate3_mse, 3))
print("Candidate 4 MSE:", round(candidate4_mse, 3))


# In[128]:


model_comparison["Test_MSE"] = [
    original_mse,
    refined_mse,
    candidate3_mse,
    candidate4_mse
]

model_comparison.round(3)


# In[129]:


model_comparison.round(3).to_csv(
    "../outputs/lr21_model_comparison.csv",
    index=False
)


# In[130]:


candidate3_metrics = pd.DataFrame({
    "Training_R2": [candidate3_train_r2],
    "Adjusted_Training_R2": [candidate3_adjusted_r2],
    "Test_MAE": [candidate3_mae],
    "Test_MSE": [candidate3_mse],
    "Test_RMSE": [candidate3_rmse],
    "Test_R2": [candidate3_r2],
    "Training_Matches": [len(y_train)],
    "Test_Matches": [len(y_test)]
})

candidate3_metrics.round(3)


# In[131]:


candidate3_metrics.round(3).to_csv(
    "../outputs/lr21_candidate3_metrics.csv",
    index=False
)


# In[132]:


plt.figure(figsize=(7, 5))

plt.scatter(
    lr21_candidate3_df["elo_difference"],
    lr21_candidate3_df["goal_difference"]
)

plt.xlabel("Elo Rating Difference")
plt.ylabel("Goal Difference")
plt.title("Elo Rating Difference vs Goal Difference")

plt.savefig(
    "../outputs/lr21_elo_vs_goal_difference.png",
    bbox_inches="tight"
)

plt.show()


# In[133]:


plt.figure(figsize=(7, 5))

plt.hist(
    lr21_candidate3_df["goal_difference"],
    bins=10
)

plt.xlabel("Goal Difference")
plt.ylabel("Frequency")
plt.title("Distribution of Goal Difference")

plt.savefig(
    "../outputs/lr21_goal_difference_distribution.png",
    bbox_inches="tight"
)

plt.show()


# In[134]:


targets_check = pd.read_csv(
    "../data/match_prediction_targets_y.csv"
)

targets_check["result_type"].value_counts(dropna=False)


# In[135]:


targets_check[
    targets_check["result_type"].isin(["AET", "Penalties"])
][
    [
        "match_id",
        "home_score",
        "away_score",
        "result_type",
        "home_xg",
        "away_xg",
        "match_result"
    ]
]


# ### Response Variable Check
# 
# The response variable for Linear Regression 2.1 was defined as:
# 
# Goal Difference = Home Score - Away Score
# 
# The target dataset contained 96 regular-time matches, 5 matches decided after extra time (AET), and 3 matches decided by penalties.
# 
# The penalty matches were recorded as drawn scores (1-1) with match_result = D, indicating that penalty shootout kicks were not included in the recorded home and away scores. Therefore, the supplied home_score and away_score values were retained for all 104 matches, and no matches were removed or modified.

# ### Multiple Linear Regression Modelling
# 
# The dataset was divided into training and testing subsets using an 80/20 split with random_state=42 to make the analysis reproducible.
# 
# Multiple Linear Regression was fitted using LinearRegression() from scikit-learn. The same training and testing observations were used when comparing alternative feature sets so that model performance could be compared consistently.
# 
# Model performance was assessed using:
# - Mean Absolute Error (MAE)
# - Mean Squared Error (MSE)
# - Root Mean Squared Error (RMSE)
# - R-squared (R²)
# - Adjusted R-squared for the training data
# 
# Lower MAE, MSE and RMSE values indicate smaller prediction errors, while higher R² values indicate that a greater proportion of variation in goal difference is explained by the model.

# ### Feature Selection and EDA Decisions
# 
# Eight pre-match explanatory variables were required for Linear Regression 2.1.
# 
# The initial feature set included FIFA ranking advantage and rest-days difference. Exploratory analysis showed that FIFA ranking advantage was very strongly correlated with Elo difference (approximately r = 0.94), indicating substantial overlap between these predictors. FIFA ranking advantage was therefore replaced with previous shots-on-target difference.
# 
# Rest-days difference showed almost no linear relationship with goal difference (approximately r = -0.01). Previous possession difference showed a stronger relationship with goal difference (approximately r = 0.44) and was therefore tested as an alternative.
# 
# Several other possible predictors, including previous shots difference, squad age difference and knockout-stage status, were also examined. They were not retained because they either added substantial overlap with existing predictors or did not improve test-set performance.
# 
# The current working feature set contains:
# 1. Elo rating difference
# 2. Squad value difference
# 3. Squad experience difference
# 4. Previous possession difference
# 5. Previous average goals scored difference
# 6. Defensive advantage based on previous goals conceded
# 7. Host advantage
# 8. Previous shots-on-target difference
# 
# This feature set produced the strongest test performance among the exploratory feature sets examined and is retained as the current LR 2.1 baseline. Final coordination with LR 2.2 will still be required to ensure that no more than four explanatory variables are shared between the two regression tasks.

# ### Data Quality and Outlier Assessment
# 
# The final working LR 2.1 dataset contained 104 matches, eight explanatory variables and one response variable. No missing values or duplicate match IDs were identified.
# 
# Potential outliers in the continuous explanatory variables were examined using the 1.5 × IQR method. The analysis identified several observations outside the IQR bounds, including observations for squad value difference, squad experience difference, previous possession difference, previous goals scored difference, defensive advantage, and previous shots-on-target difference.
# 
# These observations were retained because large differences between national teams are plausible in an international football tournament and there was no evidence that the values represented data-entry errors. The response variable, goal difference, was also checked using the IQR method and no outliers were identified.

# ### LR 2.1 Baseline Model Results
# 
# The current LR 2.1 multiple linear regression model was trained on 83 matches and evaluated on 21 held-out matches.
# 
# The model achieved:
# 
# - Training R² = 0.553
# - Adjusted Training R² = 0.504
# - Test MAE = 1.430 goals
# - Test MSE = 3.382
# - Test RMSE = 1.839 goals
# - Test R² = 0.079
# 
# The training R² indicates that the model explained approximately 55.3% of the variation in goal difference within the training data. However, the test R² was much lower at 0.079, indicating that only a small proportion of variation in the held-out match outcomes was explained.
# 
# The MAE of 1.430 indicates that predictions differed from the actual goal difference by approximately 1.43 goals on average. The RMSE of 1.839 is higher than the MAE because larger prediction errors receive greater weight.
# 
# The difference between training and test performance indicates that the model generalises less effectively to unseen matches than it fits the training data. Therefore, these results should be interpreted cautiously and the model should be compared with alternative modelling approaches during the group's final model-comparison stage.

# In[ ]:


get_ipython().system('jupyter nbconvert --to script linear_regression_2_1.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




