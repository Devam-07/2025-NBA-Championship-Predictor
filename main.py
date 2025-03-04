# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:10:02 2025

@author: devam
"""
import nba_api
from nba_api.stats.endpoints import PlayerDashPtShotDefend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import AllTimeLeadersGrids
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import cumestatsplayer
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamestimatedmetrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
#%%
all_season_data = []

#%%
team_metrics = team_metrics.sort_values('TEAM_ID')
team_stats = team_stats.sort_values("TEAM_ID")
team_metrics.reset_index(inplace=True)
team_stats.reset_index(inplace=True)
#%%
team_metrics.drop('index', axis=1, inplace=True)
team_stats.drop('index', axis=1, inplace=True)
team_stats[['OFF_RATING', 'DEF_RATING', 'NET_RATING']] = team_metrics[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING']]
#%%
champ = []
for i in range(0,30):
    champ.append(0)
team_stats["championship"] = champ
#%%
team_stats.loc[1,'championship'] = 1
#%%
import time 

all_seasons_data = []
seasons = ["2010-11", "2011-12", "2012-13", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
df = pd.DataFrame()
for season in seasons: 
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season).get_data_frames()[0]
        team_stats = team_stats[["TEAM_ID", "TEAM_NAME", "W", "L", "W_PCT"]]
        team_metrics = team_metrics.sort_values('TEAM_ID')
        team_stats = team_stats.sort_values("TEAM_ID")
        team_metrics.reset_index(inplace=True)
        team_stats.reset_index(inplace=True)
        team_metrics.drop('index', axis=1, inplace=True)
        team_stats.drop('index', axis=1, inplace=True)
        team_stats[['OFF_RATING', 'DEF_RATING', 'NET_RATING']] = team_metrics[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING']]
        team_stats['SEASON'] = season
        team_stats['CHAMPIONSHIP'] = 0
        all_seasons_data.append(team_stats)
        time.sleep(1)
    except Exception as e:
        print(f"Failed to fetch {season} data because of {e}")
df = pd.concat(all_seasons_data, ignore_index=True)

champions = {
    "2010-11": "Dallas Mavericks",
    "2011-12": "Miami Heat",
    "2012-13": "Miami Heat",
    "2013-14": "San Antonio Spurs",
    "2014-15": "Golden State Warriors",
    "2015-16": "Cleveland Cavaliers",
    "2016-17": "Golden State Warriors",
    "2017-18": "Golden State Warriors",
    "2018-19": "Toronto Raptors",
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets"
    }
        
df["CHAMPIONSHIP"] = df.apply(lambda row: 1 if row["TEAM_NAME"] == champions.get(row["SEASON"], "") else 0, axis=1)
#%%
X = np.asarray(df[['W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']])
y = np.asarray(df['CHAMPIONSHIP'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from xgboost import XGBClassifier
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

grid = GridSearchCV(XGBClassifier(random_state=42), params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)
#%%
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.7, random_state=42)
model.fit(X_train, y_train)
#%%
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.19703409075737).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#%%
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal Decision Threshold: {best_threshold}")

# Plot Precision-Recall Curve
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(x=best_threshold, color='r', linestyle='--', label="Optimal Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Optimal Probability Threshold for Deciding Champion")
plt.show()
#%%

team_metrics_c = teamestimatedmetrics.TeamEstimatedMetrics(season="2024-25").get_data_frames()[0]
team_stats_c = leaguedashteamstats.LeagueDashTeamStats(season="2024-25").get_data_frames()[0]
team_stats_c = team_stats_c[["TEAM_ID", "TEAM_NAME", "W", "L", "W_PCT"]]
team_metrics_c = team_metrics_c.sort_values('TEAM_ID')
team_stats_c = team_stats_c.sort_values("TEAM_ID")
team_metrics_c.reset_index(inplace=True)
team_stats_c.reset_index(inplace=True)
team_metrics_c.drop('index', axis=1, inplace=True)
team_stats_c.drop('index', axis=1, inplace=True)
team_stats_c[['OFF_RATING', 'DEF_RATING', 'NET_RATING']] = team_metrics_c[['E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING']]
#%%
X_c = np.asarray(team_stats_c[['W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']])
X_c = preprocessing.StandardScaler().fit(X_c).transform(X_c)
#%%
y_pred_c = model.predict(X_c)
y_pred_proba_c = model.predict_proba(X_c)[:, 1]
y_pred_c = (y_pred_proba_c > 0.19703409075737).astype(int)

#%%
plt.bar(team_stats_c['TEAM_NAME'], y_pred_proba_c)
plt.xticks(rotation=90)
plt.title("Probability for Each Team to Win the Championship")
plt.xlabel("Team")
plt.ylabel("Probability")

