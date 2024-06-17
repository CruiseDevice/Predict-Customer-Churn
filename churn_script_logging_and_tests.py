'''
Churn library testing script

This script contains functions to test the functionalities of the Churn
Library. It logs the results of the tests to a log file for review and
debugging purposes.

Usage:
    Run this script to execute all tests:
    ```sh
    python churn_script_logging_and_tests.py
    ```

Author: Akash Chavan
Creation Date: 14/06/2024
'''


import logging
import pytest
import pandas as pd
import churn_library as cls
import os

if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope='module')
def data_frame():
    '''
    Fixture for loading data.
    '''
    df = cls.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )

    return df


@pytest.fixture(scope='module')
def encoded_data_frame(data_frame):
    """
    Fixture for encoding categorical columns.
    """
    categorical_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]
    encoded_df = cls.encoder_helper(data_frame, categorical_columns, 'Churn')
    return encoded_df


@pytest.fixture(scope='module')
def split_data(encoded_data_frame):
    """
    Fixture for splitting data into training and test sets.
    """
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(encoded_data_frame, 'Churn')
    return X_train, X_test, y_train, y_test


def test_import(data_frame):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        assert isinstance(data_frame, pd.DataFrame), "Data is not a DataFrame"
        assert not data_frame.empty, "DataFrame is empty"
        logging.info("Testing import_data: SUCCESS")
    except Exception as err:
        logging.error(f"Testing imoprt_data: ERROR - {err}")
        raise err


def test_eda(data_frame):
    '''
    test perform eda function
    test creation of figures saved while performing eda
    '''
    try:
        cls.perform_eda(data_frame)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error(f"Testing perform_eda: ERROR - {err}")
        raise err

    image_name_list = [
        'churn.png', 'cr_matrix.png', 'customer_age.png',
        'marital_status.png', 'total_trans_ct.png'
    ]

    for image_name in image_name_list:
        try:
            with open(f'./images/eda/{image_name}', 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                f"Testing perform_data: image file {image_name} missing")
            raise err


def test_encoder_helper(encoded_data_frame):
    '''
    test encoder helper
    '''
    try:
        for column in ["Gender_Churn",
                       "Education_Level_Churn",
                       "Marital_Status_Churn",
                       "Income_Category_Churn",
                       "Card_Category_Churn"]:
            assert column in encoded_data_frame.columns, f"{column}_Churn is not in DataFrame"
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing encoder_helper: ERROR - {err}")
        raise err

    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(split_data):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = split_data
    try:
        assert len(X_train) == len(y_train), "Mismatch in training data"
        assert len(X_test) == len(y_test), "Mismatch in testing data"
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: Error - {err}")
        raise err


def test_train_models(split_data):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = split_data
    try:
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = cls.train_models(
            X_train, X_test, y_train, y_test)
        assert y_train_preds_rf.shape == y_train.shape, \
            "Mismatch in training predictions for Random Forest"
        assert y_test_preds_rf.shape == y_test.shape, \
            "Mismatch in testing prediction for Random Forest"
        assert y_train_preds_lr.shape == y_train.shape, \
            "Mismatch in training prediction for Logistic Regression"
        assert y_test_preds_lr.shape == y_test.shape, \
            "Mismatch in testing prediction for Logistic Regression"

        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: ERROR - {err}")
        raise err


if __name__ == "__main__":
    pytest.main()
