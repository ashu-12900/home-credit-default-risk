import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = r"C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk\application_train.csv"

def load_data(path=data):
    df = pd.read_csv(path)
    return df

def feature_engineering(df):
    # Credit ratios
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']      / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']     / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM']          = df['AMT_ANNUITY']     / df['AMT_CREDIT']
    df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    # Age and employment
    df['AGE_YEARS']         = df['DAYS_BIRTH']    / -365
    df['EMPLOYED_YEARS']    = df['DAYS_EMPLOYED'].clip(upper=0) / -365
    df['EMPLOYED_TO_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)

    # DPD ratio features — strongest predictors of default
    df['BUREAU_DEBT_CREDIT_RATIO'] = df['BUREAU_AMT_DEBT_MEAN']  / (df['BUREAU_AMT_CREDIT_MEAN'].abs()  + 1)
    df['INST_PAYMENT_RATIO']       = df['INST_AMT_PAYMENT_MEAN'] / (df['INST_AMT_DIFF_MEAN'].abs()       + 1)
    df['CC_DPD_RATIO']             = df['CC_SK_DPD_MEAN']        / (df['CC_AMT_BALANCE_MEAN'].abs()      + 1)
    df['POS_DPD_RATIO']            = df['POS_SK_DPD_MEAN']       / (df['POS_CNT_INSTALMENT_MEAN']        + 1)

    return df

def clean_data(df, remove_outliers=False, fix_skew=False):
    # Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"missing count": missing, "missing %": missing_percent})
    cols_drop = missing_df[missing_df["missing %"] > 50].index
    df = df.drop(cols_drop, axis=1, errors="ignore")

    # Fill missing
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove outliers using IQR
    # if remove_outliers:
    #     num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    #     num_cols = [col for col in num_cols if col != "TARGET"]  # exclude TARGET
    #     for col in num_cols:
    #         Q1 = df[col].quantile(0.25)
    #         Q3 = df[col].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower = Q1 - 1.5 * IQR
    #         upper = Q3 + 1.5 * IQR
    #         df[col] = df[col].clip(lower=lower, upper=upper)

    # Fix both right and left skew
    # if fix_skew:
    #     num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    #     num_cols = [col for col in num_cols if col != "TARGET"]  # exclude TARGET
    #     for col in num_cols:
    #         if df[col].skew() > 1:                          # right skew
    #             df[col] = np.log1p(df[col].clip(lower=0))
    #         elif df[col].skew() < -1:                       # left skew
    #             df[col] = np.log1p(df[col].max() - df[col])

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns
    binary_cols, multi_cols = [], []
    for col in cat_cols:
        if df[col].nunique() == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def split_data(df):
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.2
    )
    print("CLEANED AND SPLIT DATA")
    return X_train, X_test, y_train, y_test

def feature_selection(X_train, y_train, X_test, top_n=70):
    mi = mutual_info_classif(X_train, y_train)
    mi_scores = pd.Series(mi, index=X_train.columns)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = pd.Series(rf.feature_importances_, index=X_train.columns)

    feature_scores = pd.DataFrame({'MI': mi_scores, 'RF': rf_scores})
    feature_scores['MI_NORM'] = feature_scores['MI'] / feature_scores['MI'].max()
    feature_scores['RF_NORM'] = feature_scores['RF'] / feature_scores['RF'].max()
    feature_scores['FINAL'] = (feature_scores['MI_NORM'] + feature_scores['RF_NORM']) / 2
    feature_scores = feature_scores.sort_values(by='FINAL', ascending=False)

    selected_features = feature_scores.head(top_n).index.tolist()
    return X_train[selected_features], X_test[selected_features], selected_features