"""
This project aims to analyze data related to traffic accidents 2022 in England
focusing on identifying patterns and key contributing factors.
Dataset source:
https://www.kaggle.com/datasets/juhibhojani/road-accidents-data-2022
"""
import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import labels
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2_contingency
from keras.models import Sequential
from keras.layers import Dense

pd.options.display.max_columns = 200


def load_csv(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    return pd.read_csv(file_path)


def initial_investigation(df):
    # Exploring the dataframe
    print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df.head())
    print('List of column names: \n', df.columns)
    print('List of data types: ', df.dtypes)
    print(df.describe())


def df_clean(df):
    filtered_df = df.drop([
        'accident_index',
        'accident_year',
        'accident_reference',
        'casualty_reference',
        'status', 'age_of_casualty',
        'vehicle_reference',
        'lsoa_of_casualty'
        ], axis=1)

    print(f"The number of Nan values: {filtered_df.isna().sum()}")
    print(f"The number of null values: {filtered_df.isnull().sum()}")
    print(f"The dataset has {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns.")
    print('List of column names: \n', filtered_df.columns)
    print('List of data types: ', filtered_df.dtypes)    # object dtype means string

    # Delete rows with NaN values
    filtered_df = filtered_df.dropna()
    print(f"The dataset has {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns after removing Nans.")

    # Delete rows with -1 values (unknown values)
    num_rows_with_minus_one = filtered_df.isin([-1]).any(axis=1).sum()
    print(f"Number of total rows with -1: {num_rows_with_minus_one}")

    columns = filtered_df.columns
    for col in columns:
        filtered_df = filtered_df.drop(filtered_df[filtered_df[col] == -1].index)
    filtered_df = filtered_df.reset_index()
    # General information about cleaned data
    print(f"After data cleaning the dataset has {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns.")
    print(filtered_df.describe())
    print('The count of unique values in each column:')
    print(filtered_df.nunique())
    return filtered_df


def age_count_plt(df):
    # Plot the distribution of accidents by age band
    sns.countplot(x='age_band_of_casualty', data=df, palette='gist_rainbow')
    plt.title('Age band of Casualty Distribution')
    plt.xlabel('Age band of Casualty')
    plt.ylabel('Count')
    plt.legend(title='age_band_of_casualty', labels=list(labels.age_band_of_casualty.values()))
    plt.show()

    # Form a dataframe only for pedestrian
    # 'Drop=true' drops the previous index
    pedestrian_df = df[df['pedestrian_location'] != 0].reset_index(drop=True)

    # Plot the distribution of accidents by age band for pedestrian
    sns.countplot(x='age_band_of_casualty', data=pedestrian_df, palette='gist_rainbow')
    plt.title('Age band of Casualty Distribution in Pedestrian')
    plt.xlabel('Age band of Casualty')
    plt.ylabel('Count')
    plt.legend(title='age_band_of_casualty', labels=list(labels.age_band_of_casualty.values()), loc='upper right')
    plt.show()


def gender_count_plt(df):
    # Plot the distribution of accidents by gender
    sns.countplot(x='sex_of_casualty', data=df, palette='gist_rainbow')
    plt.title('Distribution of Accidents by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='sex_of_casualty', labels=list(labels.sex_of_casualty.values()))
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.show()


def area_count_plt(df):
    # Plot the distribution of accidents by casualty's home area type
    sns.countplot(x='casualty_home_area_type', data=df, palette='gist_rainbow')
    plt.title('Distribution of Accidents by Casualty Home Area Type')
    plt.xlabel('Home Area Type')
    plt.ylabel('Count')
    plt.legend(title='casualty_home_area_type', labels=list(labels.casualty_home_area_type.values()))
    plt.xticks([0, 1, 2], list(labels.casualty_home_area_type.values()))
    plt.show()


def loc_count_plt(df):
    # Plot the distribution of pedestrian location values
    sns.countplot(x='pedestrian_location', data=df, palette='gist_rainbow')
    plt.title('Distribution of Pedestrian Location Values')
    plt.xlabel('Pedestrian Location')
    plt.ylabel('Count')
    plt.xticks(list(labels.pedestrian_location.keys()), list(labels.pedestrian_location.values()), rotation=90)
    plt.show()


def mov_count_plt(df):
    # Plot the distribution of pedestrian movement values
    sns.countplot(x='pedestrian_movement', data=df, palette='gist_rainbow')
    plt.title('Distribution of Pedestrian Movement Values')
    plt.xlabel('Pedestrian Movement')
    plt.ylabel('Count')
    plt.xticks(list(labels.pedestrian_movement.keys()), list(labels.pedestrian_movement.values()), rotation=90)
    plt.show()


def gender_severity_plt(df):
    # Plot the relationship between casualty severity and gender
    sns.countplot(x='casualty_severity', hue='sex_of_casualty', data=df)
    plt.title('Casualty Severity by Gender')
    plt.xlabel('Casualty Severity')
    plt.ylabel('Count')
    plt.legend(title='casualty_severity', labels=list(labels.casualty_severity.values()))
    plt.xticks([0, 1, 2], list(labels.casualty_severity.values()))
    plt.show()


def class_severity_plt(df):
    # Plot the relationship between casualty severity and casualty class
    sns.countplot(x='casualty_severity', hue='casualty_class', data=df)
    plt.title('Casualty Severity by Casualty Class')
    plt.xlabel('Casualty Severity')
    plt.ylabel('Count')
    plt.legend(title='Casualty Class', labels=list(labels.casualty_class.values()))
    plt.xticks([0, 1, 2], list(labels.casualty_severity.values()))
    plt.show()


def area_severity_plt(df):
    # Plot the relationship between casualty severity and casualty's home area type
    sns.countplot(x='casualty_severity', hue='casualty_home_area_type', data=df)
    plt.title('Casualty Severity by Home Area Type')
    plt.xlabel('Casualty Severity')
    plt.ylabel('Count')
    plt.legend(title='Home Area Type', labels=['Urban', 'Small Town', 'Rural'])
    plt.xticks([0, 1, 2], list(labels.casualty_severity.values()))
    plt.show()


def type_count_plt(df):
    # Plot the distribution of casualty types
    sns.countplot(x='casualty_type', data=df, palette='gist_rainbow')
    plt.title('Distribution of Casualty Types')
    plt.xlabel('Casualty Type')
    plt.ylabel('Count')
    x_ticks = list(range(21))
    print(x_ticks)
    plt.xticks(list(range(21)), list(labels.casualty_type.values()), rotation=90)
    plt.show()


def group_by(df, column_name):
    grouped_data = df.groupby(column_name)['casualty_severity'].count()
    sorted_data = grouped_data.sort_values(ascending=False)
    print(sorted_data)


def feature_target_dependency(df, feature_columns):
    # Evaluating the dependency of casualty severity to features
    for column in feature_columns:
        contingency_table = pd.crosstab(df['casualty_severity'], df[column])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        # p_value under 0.05 suggest that these variables are related in some way.
        # print(f"Chi-Square Statistic: {chi2_stat:.4f}")
        print(f"P-value of {column} is : {p_value:.10f}")


def data_split(df):
    # Forming a dataframe to train models
    model_df = df[main_columns]
    x = model_df.drop(columns=['casualty_severity'])  # Features
    y = model_df['casualty_severity']  # Target
    # Set the train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    return model_df, x, y, x_train, x_test, y_train, y_test


def d3(x_train, x_test, y_train, y_test, feature_columns):
    # Decision Tree
    model = DecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    # Make predictions
    y_pred = model.predict(x_test)
    print('\nDTree:')
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # Feature importance are normalized values that sum up to 1.
    features = pd.DataFrame(model.feature_importances_, index=feature_columns).sort_values(by=0, ascending=False)
    print(features.head(11))


def extra_trees(x_train, y_train):
    # Extra trees
    model = ExtraTreesClassifier(random_state=1)
    model.fit(x_train, y_train)
    print(cross_val_score(model, x_train, y_train, scoring='accuracy', cv=5, n_jobs=-1))


def knn(x, x_test, y, y_test, feature_columns):
    # KNN model
    model = KNeighborsClassifier()
    # fit the model
    model.fit(x, y)
    # perform permutation importance
    results = permutation_importance(model, x, y, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    print('\nKNN')
    for i, v in enumerate(importance):
        print(f"Feature: {feature_columns[i]}, Score: {v:.5f}")
    # plot feature importance
    plt.bar([k for k in range(len(importance))], importance)
    plt.show()
    accuracy = model.score(x_test, y_test)
    print(f"KNN Model accuracy: {accuracy:.2f}")


def support_vector_machine(x_train, x_test, y_train, y_test, feature_columns):
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    results = permutation_importance(model, x_test, y_test, scoring='accuracy')
    # Get importance scores
    importance_scores = results.importances_mean
    # Sort features by importance (descending order)
    sorted_indices = importance_scores.argsort()[::-1]

    # Print feature names and their importance scores
    print("Feature Importance:")
    for idx in sorted_indices:
        print(f"{feature_columns[idx]}: {importance_scores[idx]:.5f}")


def rnd_forest(x_train, x_test, y_train, y_test, feature_columns):
    # Random Forest
    model = RandomForestClassifier(n_estimators=5, random_state=1)
    model.fit(x_train, y_train)
    importance = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
    # Sort the DataFrame by importance in descending order
    sorted_feature_df = feature_df.sort_values(by='Importance', ascending=False)
    print('\n Random Forest:')
    # summarize feature importance
    # Print the sorted feature importance
    for i, row in sorted_feature_df.iterrows():
        print(f"Feature: {row['Feature']}, Score: {row['Importance']:.5f}")
    # plot feature importance
    plt.bar(sorted_feature_df['Feature'], sorted_feature_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.show()
    print("Accuracy on test set: {:.3f}".format(model.score(x_test, y_test)))


def neural_network(model_df, main_columns):
    # Neural Network model
    # Encode nominal features (one-hot encoding)
    df_encoded = pd.get_dummies(model_df, columns=main_columns)
    # Split data
    X_nn = df_encoded.drop(columns=['casualty_severity_1', 'casualty_severity_2', 'casualty_severity_3'])
    y_nn = df_encoded[['casualty_severity_1', 'casualty_severity_2', 'casualty_severity_3']]
    X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X_nn, y_nn, test_size=0.2, random_state=1)
    # Create a neural network model
    model = Sequential()
    model.add(Dense(units=59, activation='relu', input_dim=X_nn_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))  # Output layer for 3 classes

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_nn_train, y_nn_train, epochs=10, batch_size=32, validation_data=(X_nn_test, y_nn_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_nn_test, y_nn_test)
    print(f"Test accuracy: {accuracy:.2f}")


def bayes(x_train, x_test, y_train, y_test, feature_columns):
    # Bayesian model
    label_encoders = {}
    for feature in feature_columns:
        le = LabelEncoder()
        model_df[feature] = le.fit_transform(model_df[feature])
        label_encoders[feature] = le
    model = CategoricalNB()
    model.fit(x_train, y_train)
    # Make predictions
    y_pred = model.predict(x_test)
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))


def rnd_forest_grd(model_df, feature_columns):
    # Random Forest and GridSearchCV
    label_encoders = {}
    for feature in feature_columns:
        le = LabelEncoder()
        model_df[feature] = le.fit_transform(model_df[feature])
        label_encoders[feature] = le
    # Split data into train and test sets
    X_grd = model_df[feature_columns]
    y_grd = model_df['casualty_severity']
    X_grd_train, X_grd_test, y_grd_train, y_grd_test = train_test_split(X_grd, y_grd, test_size=0.2, random_state=42)
    # Create classifier
    rfc = RandomForestClassifier(random_state=42)
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Fit the model
    grid_search.fit(X_grd_train, y_grd_train)
    # Get the best estimator
    model = grid_search.best_estimator_
    # Make predictions
    y_pred = model.predict(X_grd_test)
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_grd_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_grd_test, y_pred))


def d3_grd(model_df, feature_columns):
    # DecisionTreeClassifier and GridSearchCV
    label_encoders = {}
    for feature in feature_columns:
        le = LabelEncoder()
        model_df[feature] = le.fit_transform(model_df[feature])
        label_encoders[feature] = le

    # Split data into train and test sets
    X_grd = model_df[feature_columns]
    y_grd = model_df['casualty_severity']
    X_grd_train, X_grd_test, y_grd_train, y_grd_test = train_test_split(X_grd, y_grd, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier(random_state=42)
    # Define the hyperparameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5)
    # Fit the model
    grid_search.fit(X_grd_train, y_grd_train)
    # Get the best estimator
    model = grid_search.best_estimator_
    # Make predictions
    y_pred = model.predict(X_grd_test)
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_grd_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_grd_test, y_pred))

    features = pd.DataFrame(model.feature_importances_, index=feature_columns).sort_values(by=0, ascending=False)
    print(features.head(11))


# Load the dataframe
df = load_csv(f'accidents.csv')

# Show some general information about dataframe
initial_investigation(df)

# Cleaning the dataframe
df = df_clean(df)

# Plot some important charts
age_count_plt(df)
gender_count_plt(df)
area_count_plt(df)
loc_count_plt(df)
mov_count_plt(df)
gender_severity_plt(df)
class_severity_plt(df)
area_severity_plt(df)
type_count_plt(df)

# This data shows that just group 0 in casualty_type are pedestrian
# And other groups are somehow in/on a vehicle
group_by(df, 'casualty_imd_decile')

# Forming the lists of required columns
main_columns = list(df.drop('index', axis=1).columns)
print('Main column names: ', main_columns)
feature_columns = list(df.drop(['index', 'casualty_severity'], axis=1).columns)
print('Feature column names: ', feature_columns)

model_df, X, y, X_train, X_test, y_train, y_test = data_split(df)

d3(X_train, X_test, y_train, y_test, feature_columns)
extra_trees(X_train, y_train)
#knn(X, X_test, y, y_test, feature_columns)
#support_vector_machine(X_train, X_test, y_train, y_test, feature_columns)
rnd_forest(X_train, X_test, y_train, y_test, feature_columns)
#neural_network(model_df, main_columns)
bayes(X_train, X_test, y_train, y_test, feature_columns)
#rnd_forest_grd(model_df, feature_columns)
d3_grd(model_df, feature_columns)


# Indicating the most effective factors on casualty severity
group_by(df, 'casualty_type')
group_by(df, 'age_band_of_casualty')
group_by(df, 'casualty_imd_decile')
