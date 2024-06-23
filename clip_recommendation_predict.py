import pandas as pd
import numpy as np
from tabulate import tabulate
from surprise import Dataset, Reader
from sklearn.preprocessing import MinMaxScaler
import joblib


# select number of recommendations to the user
n_recommendations = 10

# Load the trained model
model = joblib.load('best_model_SVD.pkl')

# Load and preprocess training data (for scaling purposes)
df_train = pd.read_csv('scenes_emotional_ratings.csv')
df_train = df_train.dropna(how='any')
df_train = df_train[df_train['emotional_rating'] != 0]

# Load test data for the new user
df_test = pd.read_csv('scenes_emotional_ratings_test.csv')
df_test = df_test.dropna(how='any')
df_test = df_test[df_test['emotional_rating'] != 0]

# Scale emotional_rating and RT_rating to [0, 1] using the training data for fitting
scaler = MinMaxScaler()
scaler.fit(df_train[['emotional_rating', 'rating_RT']])

# Transform both columns together
scaled_features = scaler.transform(df_test[['emotional_rating', 'rating_RT']])
df_test['emotional_rating_scaled'] = scaled_features[:, 0]
df_test['rating_RT_scaled'] = scaled_features[:, 1]

# Invert the scaled reaction time (1 - RT_scaled) so that lower RT contributes more
df_test['inverted_RT_scaled'] = 1 - df_test['rating_RT_scaled']

# Combine the scaled ratings with given weights
df_test['combined_rating'] = 0.8 * df_test['emotional_rating_scaled'] + 0.2 * df_test['inverted_RT_scaled']

# Get the new user ID
new_user_id = df_test['ID'].iloc[0]

# Get all possible clips
all_possible_clips = df_train['clip_item'].unique()

# Get the clips rated by the new user
user_rated_clips = df_test['clip_item'].unique()


# Function to recommend top clips for a given user
def recommend_clips(model, user_id, all_possible_clips, user_rated_clips, n_recommendations=n_recommendations):
    clip_items_to_predict = [clip for clip in all_possible_clips if clip not in user_rated_clips]
    predictions = [model.predict(user_id, clip_item) for clip_item in clip_items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_predictions = predictions[:n_recommendations]
    return [(pred.iid, pred.est) for pred in top_n_predictions]


# Get recommendations for the new user
recommended_clips = recommend_clips(model, new_user_id, all_possible_clips, user_rated_clips)

# Print the new user's ratings
print(f"Ratings provided by user {new_user_id}:")
user_ratings_table = df_test[['clip_item', 'combined_rating']].values.tolist()
print(tabulate(user_ratings_table, headers=['Clip', 'Combined Rating'], tablefmt='pretty'))

# Print recommendations
print(f"\nTop 10 recommended clips for user {new_user_id} using SVD:")
recommendations_table = [(i, clip, f"{rating:.4f}") for i, (clip, rating) in enumerate(recommended_clips, 1)]
print(tabulate(recommendations_table, headers=['Rank', 'Clip', 'Predicted Rating'], tablefmt='pretty'))

# Additional analysis
print("\nRecommendation Analysis:")
print(f"Number of unique ratings in top 10: {len(set([rating for _, rating in recommended_clips]))}")
print(f"Range of ratings: {min([rating for _, rating in recommended_clips]):.4f} - {max([rating for _, rating in recommended_clips]):.4f}")

# Calculate and print the average reaction time for recommended clips
recommended_clip_ids = [clip for clip, _ in recommended_clips]
avg_rt_recommended = df_train[df_train['clip_item'].isin(recommended_clip_ids)]['rating_RT'].mean()
print(f"Average reaction time for recommended clips: {avg_rt_recommended:.4f}")

# Calculate and print the average reaction time for all clips
avg_rt_all = df_train['rating_RT'].mean()
print(f"Average reaction time for all clips: {avg_rt_all:.4f}")

# Compare the two
if avg_rt_recommended < avg_rt_all:
    print("The recommended clips have a lower average reaction time, suggesting higher confidence in ratings.")
else:
    print("The recommended clips have a higher average reaction time, suggesting lower confidence in ratings.")