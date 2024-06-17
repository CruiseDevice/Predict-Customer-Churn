"""
Churn Library

This module contains functions for loading data, performing exploratory data
analysis, encoding categorical features, engineering features, training models,
and evaluating model performance.

Functions:
    import_data(pth): Loads the data from the specified file path.
    perform_eda(data_frame): Performs exploratory data analysis and saves plots.
    encoder_helper(data_frame, category_lst, response): Encodes categorical features.
    perform_feature_engineering(data_frame, response): Splits data into training and
                                               test sets.
    train_models(X_train, X_test, y_train, y_test): Trains RandomForest and
                                                    Logistic Regression models.
"""

# import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, plot_roc_curve

import joblib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''

    # Select only numerical columns
    df_numeric = data_frame.select_dtypes(include=['number'])

    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.title('Histogram of Churn')
    plt.savefig('./images/eda/churn.png')

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.title('Histogram of Customer Age')
    plt.savefig('./images/eda/customer_age.png')

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Bar plot of Marital Status')
    plt.savefig('./images/eda/marital_status.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Histplot of Total_Trans_Ct')
    plt.savefig('./images/eda/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_numeric.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heatmap of Correlation Matrix')
    plt.savefig('./images/eda/cr_matrix.png')


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        new_column_name = f"{category}_{response}"
        category_mean = data_frame.groupby(category)[response].mean()
        data_frame[new_column_name] = data_frame[category].map(category_mean)

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y = data_frame[response]
    X = pd.DataFrame()
    X[keep_columns] = data_frame[keep_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_classification_report(report, title):
    """
    Plot classification report as a heatmap
    """
    lines = report.split('\n')
    report_data = []
    for line in lines[2:]:
        if line.strip() == "":
            break
        row_data = line.split()
        if len(row_data) < 5:
            continue
        row = {
            'class': row_data[0],
            'precision': float(row_data[1]),
            'recall': float(row_data[2]),
            'f1_score': float(row_data[3]),
            'support': int(row_data[4])
        }
        report_data.append(row)

    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.set_index('class', inplace=True)

    sns.heatmap(dataframe.iloc[:, :-1].astype(float),
                annot=True, cmap='Blues', fmt='.2f')


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    report_train_lr = classification_report(y_train, y_train_preds_lr)
    report_test_lr = classification_report(y_test, y_test_preds_lr)

    report_train_rf = classification_report(y_train, y_train_preds_rf)
    report_test_rf = classification_report(y_test, y_test_preds_rf)

    images_dir = 'images/results'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.figure(figsize=(10, 6))
    plot_classification_report(report_train_lr, 'Logistic Regression - Train')
    plt.savefig(
        os.path.join(
            images_dir,
            'logistic_regression_train_report.png'))

    plt.figure(figsize=(10, 6))
    plot_classification_report(report_test_lr, 'Logistic Regression - Test')
    plt.savefig(
        os.path.join(
            images_dir,
            'logistic_regression_test_report.png'))

    plt.figure(figsize=(10, 6))
    plot_classification_report(report_train_rf, 'Random Forest - Train')
    plt.savefig(os.path.join(images_dir, 'random_forest_train_report.png'))

    plt.figure(figsize=(10, 6))
    plot_classification_report(report_test_rf, 'Random Forest - Test')
    plt.savefig(os.path.join(images_dir, 'random_forest_test_report.png'))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    images_dir = 'images/results'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # get feature importances from model
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 6))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    plt.savefig(os.path.join(images_dir, 'feature_importance.png'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # plot roc_curve
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('./images/results/roc_curve.png')

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":

    # Import data
    df = import_data('./data/bank_data.csv')
    # df_sample = df.sample(frac=0.1)

    # Perform EDA
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )

    perform_eda(df)

    # Encode categorical columns
    categorical_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    encoder_helper(df, categorical_columns, 'Churn')

    # Perform Feature Engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # train models
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
        X_train, X_test, y_train, y_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Load the saved model
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Calculate feature importances
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    feature_importance_plot(rfc_model, X, './images/results/')
