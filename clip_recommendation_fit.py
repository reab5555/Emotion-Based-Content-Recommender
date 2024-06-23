import pandas as pd
import numpy as np
from tabulate import tabulate
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, cross_validate
from surprise import accuracy
from sklearn.preprocessing import MinMaxScaler
import joblib

def display_df_with_count(df, unique_ids, n=5):
    if len(unique_ids) <= n:
        display_df = df.loc[df[['ID', 'clip_item']].isin(unique_ids).all(axis=1)].copy()
    else:
        middle_index = len(unique_ids) // 2
        half_n = n // 2
        start_index = max(0, middle_index - half_n)
        end_index = min(len(unique_ids), middle_index + half_n + (1 if n % 2 != 0 else 0))
        selected_ids = unique_ids.iloc[start_index:end_index]
        display_df = df.loc[df[['ID', 'clip_item']].isin(selected_ids).all(axis=1)].copy()
        if start_index > 0:
            prev_count_row = pd.DataFrame(
                [['... and {} previous rows'.format(start_index)] + ['...'] * (len(df.columns) - 1)],
                columns=df.columns)
            display_df = pd.concat([prev_count_row, display_df], ignore_index=True)
        if end_index < len(unique_ids):
            next_count_row = pd.DataFrame(
                [['... and {} more rows'.format(len(unique_ids) - end_index)] + ['...'] * (len(df.columns) - 1)],
                columns=df.columns)
            display_df = pd.concat([display_df, next_count_row], ignore_index=True)
    display_df = display_df.map(lambda x: f'{x:.3f}'.rstrip('0').rstrip('.') if isinstance(x, (float, int)) else x)
    print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False))

# Load and preprocess data
df = pd.read_csv('scenes_emotional_ratings.csv')
df = df.dropna(how='any')
df = df[df['emotional_rating'] != 0]

# Get unique IDs and clip items
unique_ids = df[['ID', 'clip_item']].drop_duplicates()

print("Original Data:")
display_df_with_count(df, unique_ids)
print("\n")

# Scale emotional_rating and RT_rating to [0, 1]
scaler = MinMaxScaler()
df['emotional_rating_scaled'] = scaler.fit_transform(df[['emotional_rating']])
df['rating_RT_scaled'] = scaler.fit_transform(df[['rating_RT']])

# Invert the scaled reaction time (1 - RT_scaled) so that lower RT contributes more
df['inverted_RT_scaled'] = 1 - df['rating_RT_scaled']

# Combine the scaled ratings with given weights
df['combined_rating'] = 0.8 * df['emotional_rating_scaled'] + 0.2 * df['inverted_RT_scaled']

df_weighted = df[['ID', 'clip_item', 'combined_rating']].copy()

print("Weighted Data (after scaling and RT inversion):")
display_df_with_count(df_weighted, unique_ids)
print("\n")

# Convert the DataFrame to a Surprise Dataset
reader = Reader(rating_scale=(df['combined_rating'].min(), df['combined_rating'].max()))
data = Dataset.load_from_df(df[['ID', 'clip_item', 'combined_rating']], reader)

# Define parameter grid for SVD
param_grid = {
    'n_factors': [5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
    'n_epochs': [50, 75, 85, 100, 125, 150, 200, 250, 300],
    'lr_all': [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005],
    'reg_all': [0.01, 0.05, 0.1, 0.2, 0.3]
}

# Perform Grid Search for SVD
print("Performing Grid Search to find optimal parameters...")
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1, joblib_verbose=1)
gs.fit(data)
best_params = gs.best_params['rmse']

# After grid search
print('~'*100)
best_params_table = [[param, value] for param, value in best_params.items()]
best_params_table.append(['Best RMSE', f"{gs.best_score['rmse']:.3f}"])
print(tabulate(best_params_table, headers=['Parameter', 'Value'], tablefmt='pretty'))
print('~'*100)

# Perform cross-validation with the best parameters
print("\nPerforming 5-fold cross-validation with optimal parameters...")
svd = SVD(**best_params)
cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('\nCross-validation results:')
cv_table = [
    ['Mean RMSE', f"{cv_results['test_rmse'].mean():.3f} (+/- {cv_results['test_rmse'].std() * 2:.3f})"],
    ['Mean MAE', f"{cv_results['test_mae'].mean():.3f} (+/- {cv_results['test_mae'].std() * 2:.3f})"]
]
print(tabulate(cv_table, headers=['Metric', 'Value'], tablefmt='pretty'))

# Train the final model on the entire dataset
print("\nTraining final model on entire dataset...")
final_model = SVD(**best_params)
final_model.fit(data.build_full_trainset())

# Save the best model
model_filename = 'best_model_SVD.pkl'
joblib.dump(final_model, model_filename)
print(f"The best model has been saved as {model_filename}")

print('~'*100)