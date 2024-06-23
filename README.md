# EmotiSVD-Recommender

## Project Overview

This project predicts clips for viewers based on their emotional reactions and reaction times. Participants rated various emotional clips on a scale from 0 to 100, with 100 indicating very emotional and 0   indicating not emotional at all. Their reaction times to these ratings were also recorded. The goal is to leverage these ratings and reaction times to recommend clips that are likely to evoke a strong  emotional response from viewers.

## Experiment and Dataset Description

In the experiment, participants watched a variety of emotional clips and rated them based on how much the clips made them feel emotional, on a scale from 0 to 100 (where 100 is very emotional and 0 is not emotional at all). Alongside these ratings, the reaction times (RT) of the participants to provide these ratings were also recorded.

The goal is to predict clips that are likely to evoke strong emotional responses from viewers by leveraging both the emotional ratings and the reaction times.   

## Features and Innovations
**Composite Rating Calculation**: Emotional ratings are scaled between 0 and 1. Reaction times are inverted and scaled similarly, with shorter reaction times contributing more to the composite score.   
**Weighted Composite Scores**: The algorithm combines the scaled emotional ratings and reaction times with specific weights to form a composite rating for each clip.   
**Optimized Parameters through Grid Search**: The project uses Grid Search to find the optimal parameters for the SVD algorithm from the Surprise library.   
**Cross-Validation for Robustness**: The model's performance is validated using 5-fold cross-validation, ensuring its robustness.   

## Model
### Singular Value Decomposition (SVD) and Matrix Factorization    

Singular Value Decomposition (SVD) is a matrix factorization technique used in collaborative filtering for recommendation systems. It decomposes the user-item interaction matrix into three matrices, capturing latent factors that explain user preferences and item characteristics.   

### Collaborative Filtering
Collaborative filtering is a method used in recommendation systems where the preferences of users are predicted based on the preferences of similar users. SVD is one such technique that helps in identifying these similarities and making recommendations.  

### Source Table

|             ID             |                clip_item                 | emotional_rating | rating_RT |
|----------------------------|------------------------------------------|------------------|-----------|
| ... and 1230 previous rows |                   ...                    |       ...        |    ...    |
|           25910            |            Titanic (1997) 114            |        59        |   3584    |
|           25910            |            Bullitt (1968) 116            |        55        |   1733    |
|           25910            |      Across The Universe (2007) 118      |        60        |   2215    |
|           25910            |    Gentlemen Prefer Blondes (1953) 12    |        62        |   2487    |
|           25910            | The Good the Bad and the Ugly (1966) 120 |        60        |   3333    |
|   ... and 1229 more rows   |                   ...                    |       ...        |    ...    |


### Weighted Table (after scaling and RT inversion)

|             ID             |                clip_item                 | combined_rating |
|----------------------------|------------------------------------------|-----------------|
| ... and 1230 previous rows |                   ...                    |       ...       |
|           25910            |            Titanic (1997) 114            |      0.663      |
|           25910            |            Bullitt (1968) 116            |      0.634      |
|           25910            |      Across The Universe (2007) 118      |      0.674      |
|           25910            |    Gentlemen Prefer Blondes (1953) 12    |      0.689      |
|           25910            | The Good the Bad and the Ugly (1966) 120 |      0.672      |
|   ... and 1229 more rows   |                   ...                    |       ...       |


### Optimal Parameters
| Parameter | Value |
|-----------|-------|
| n_factors |   8   |
| n_epochs  |  50   |
|  lr_all   | 0.005 |
|  reg_all  |  0.1  |
| Best RMSE | 0.166 |

### Cross-Validation
Performed 5-fold cross-validation with optimal parameters.   
|           | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |  Mean  |  Std   |
|-----------|--------|--------|--------|--------|--------|--------|--------|
| RMSE      | 0.1636 | 0.1646 | 0.1616 | 0.1732 | 0.1681 | 0.1662 | 0.0041 |
| MAE       | 0.1322 | 0.1326 | 0.1275 | 0.1393 | 0.1372 | 0.1338 | 0.0041 |
| Fit time  |  0.01  |  0.01  |  0.01  |  0.01  |  0.01  |  0.01  |  0.00  |
| Test time |  0.00  |  0.00  |  0.00  |  0.00  |  0.00  |  0.00  |  0.00  |

## Prediction for new user

### Top 10 Recommended Clips for User 10130 using SVD
| Rank |             Clip              | Predicted Rating |
|------|-------------------------------|------------------|
|  1   |    Come and See (1985) 15     |      0.8452      |
|  2   |    Come and See (1985) 45     |      0.8182      |
|  3   | Final Destination 3 (2006) 36 |      0.7972      |
|  4   |      Incendies (2010) 77      |      0.7928      |
|  5   |     City of God (2002) 74     |      0.7921      |
|  6   |      Incendies (2010) 78      |      0.7903      |
|  7   |     City of God (2002) 73     |      0.7796      |
|  8   | Across the Universe (2007) 13 |      0.7680      |
|  9   |    Twin Peaks 2 (2017) 66     |      0.7570      |
|  10  | Final Destination (2000) 125  |      0.7548      |


### Algorithm steps:
1. Load and preprocess the data.
2. Scale emotional ratings and reaction times.
3. Compute the composite rating/combined weights (rating 80% + inverse reaction time 20%) for each clip.
4. Use the Surprise library to perform collaborative filtering with SVD.
5. Optimize the SVD parameters using Grid Search.
6. Validate the model using cross-validation with 5 k-folds.
7. Train the final model on the entire dataset.
8. Save the trained model for future use.

This project is licensed under the MIT License. See the LICENSE file for details.
