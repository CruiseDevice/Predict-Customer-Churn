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



import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the 
    other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    test creation of figures saved while performing eda
    '''
    # df = import_data('./data/bank_data.csv')
    # perform_eda(df)
    df = cls.import_data('./data/bank_data.csv')

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )
    df.drop(columns=['Attrition_Flag'], inplace=True)

    perform_eda(df)
    image_name_list = [
        'churn.png', 'cr_matrix.png', 'customer_age.png',
        'marital_status.png', 'total_trans_ct.png'
    ]
    for image_name in image_name_list:
        try:
            with open(f'./images/eda/{image_name}', 'r'):
                logging.info("Testing perform_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                f"Testing perform_data: image file {image_name} missing")
            raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )
    df.drop(columns=['Attrition_Flag'], inplace=True)

    categorical_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]
    encoded_df = encoder_helper(df, categorical_columns, 'Churn')
    try:
        for column in ["Gender_Churn",
                       "Education_Level_Churn",
                       "Marital_Status_Churn",
                       "Income_Category_Churn",
                       "Card_Category_Churn"]:
            assert column in encoded_df
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The df does not have the right encoded column")
        raise err

    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )

    # Encode categorical columns
    categorical_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    cls.encoder_helper(df, categorical_columns, 'Churn')

    try:
        # Perform feature engineering
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')

    # Assert that the lengths of the training and test sets are as expected
        assert len(X_train) == len(
            y_train), "Mismatch in training data lengths"
        assert len(X_test) == len(y_test), "Mismatch in test data lengths"
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except KeyError as err:
        logging.error('Testing perform_feature_engineering: ERROR')
        raise err
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Data lengths do not match")


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cls.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )

    # Encode categorical columns
    categorical_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    cls.encoder_helper(df, categorical_columns, 'Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, 'Churn')
    try:
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
            X_train, X_test, y_train, y_test)
        assert y_train_preds_rf.shape == y_train.shape, "Mismatch in training predictions for Random Forest"
        assert y_test_preds_rf.shape == y_test.shape, "Mismatch in testing prediction for Random Forest"
        assert y_train_preds_lr.shape == y_train.shape, "Mismatch in training prediction for Logistic Regression"
        assert y_test_preds_lr.shape == y_test.shape, "Mismatch in testing prediction for Logistic Regression"

        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: ERROR")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
