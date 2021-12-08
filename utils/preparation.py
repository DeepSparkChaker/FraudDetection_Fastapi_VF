# -*- coding: utf-8 -*-
"""Preparation class"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import category_encoders as ce
import numpy as np
import pandas as pd
from typing import List, Optional, Union



class CountFrequencyEncoder(BaseEstimator, TransformerMixin):
    # temp = df['card1'].value_counts().to_dict()
    # df['card1_counts'] = df['card1'].map(temp)
    def __init__(
            self,
            encoding_method: str = "count",
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            keep_variable=True,
    ) -> None:

        self.encoding_method = encoding_method
        self.variables = variables
        self.keep_variable = keep_variable

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the counts or frequencies which will be used to replace the categories.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.
        y: pandas Series, default = None
            y is not needed in this encoder. You can pass y or None.
        """
        self.encoder_dict_ = {}

        # learn encoding maps
        for var in self.variables:
            if self.encoding_method == "count":
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == "frequency":
                n_obs = float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # replace categories by the learned parameters
        X = X.copy()
        for feature in self.encoder_dict_.keys():
            if self.keep_variable:
                X[feature + '_fq_enc'] = X[feature].map(self.encoder_dict_[feature])
            else:
                X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X[self.variables].to_numpy()


#################################################
class FeaturesEngineerGroup(BaseEstimator, TransformerMixin):
    def __init__(self, groupping_method="mean",
                 variables="amount",
                 groupby_variables="nameOrig"
                 ):
        self.groupping_method = groupping_method
        self.variables = variables
        self.groupby_variables = groupby_variables

    def fit(self, X, y=None):
        """
        Learn the mean or median of  amount of each client which will be used to create new feature for each unqiue client in order to undersatant thier behavior .
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
        The training dataset. Can be the entire dataframe, not just the
        variables to be transformed.
        y: pandas Series, default = None
        y is not needed in this encoder. You can pass y or None.
        """
        self.group_amount_dict_ = {}
        # df.groupby('card1')['TransactionAmt'].agg(['mean']).to_dict()
        # temp = df.groupby('card1')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_card1_mean'},axis=1)
        # df = pd.merge(df,temp,on='card1',how='left')
        # target_mean = df_train.groupby(['id1', 'id2'])['target'].mean().rename('avg')
        # df_test = df_test.join(target_mean, on=['id1', 'id2'])
        # lifeExp_per_continent = gapminder.groupby('continent').lifeExp.mean()
        # learn mean/medain
        # for groupby in self.groupby_variables:
        #   for var in self.variables:
        if self.groupping_method == "mean":
            self.group_amount_dict_[self.variables] = X.fillna(np.nan).groupby([self.groupby_variables])[
                self.variables].agg(['mean']).to_dict()
        elif self.groupping_method == "median":
            self.group_amount_dict_[self.variables] = X.fillna(np.nan).groupby([self.groupby_variables])[
                self.variables].agg(['median']).to_dict()
        else:
            print('error , chose mean or median')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # for col in self.variables:
        #   for agg_type in self.groupping_method:
        new_col_name = self.variables + '_Transaction_' + self.groupping_method
        X[new_col_name] = X[self.groupby_variables].map(self.group_amount_dict_[self.variables][self.groupping_method])
        return X[new_col_name].to_numpy().reshape(-1, 1)

    ################################################


class FeaturesEngineerGroup2(BaseEstimator, TransformerMixin):
    def __init__(self, groupping_method="mean",
                 variables="amount",
                 groupby_variables="nameOrig"
                 ):
        self.groupping_method = groupping_method
        self.variables = variables
        self.groupby_variables = groupby_variables

    def fit(self, X, y=None):
        """
        Learn the mean or median of  amount of each client which will be used to create new feature for each unqiue client in order to undersatant thier behavior .
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
        The training dataset. Can be the entire dataframe, not just the
        variables to be transformed.
        y: pandas Series, default = None
        y is not needed in this encoder. You can pass y or None.
        """
        X = X.copy()
        self.group_amount_dict_ = {}
        # df.groupby('card1')['TransactionAmt'].agg(['mean']).to_dict()
        # temp = df.groupby('card1')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_card1_mean'},axis=1)
        # df = pd.merge(df,temp,on='card1',how='left')
        # target_mean = df_train.groupby(['id1', 'id2'])['target'].mean().rename('avg')
        # df_test = df_test.join(target_mean, on=['id1', 'id2'])
        # lifeExp_per_continent = gapminder.groupby('continent').lifeExp.mean()
        # learn mean/medain
        # for groupby in self.groupby_variables:
        #   for var in self.variables:

        print('we have {} unique clients'.format(X[self.groupby_variables].nunique()))
        new_col_name = self.variables + '_Transaction_' + self.groupping_method
        X[new_col_name] = X.groupby([self.groupby_variables])[[self.variables]].transform(self.groupping_method)
        X = X.drop_duplicates(['nameOrig'])

        self.group_amount_dict_ = dict(zip(X[self.groupby_variables], X[new_col_name]))
        del X
        # print('we have {} unique mean amount : one for each client'.format(len(self.group_amount_dict_)))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # for col in self.variables:
        #   for agg_type in self.groupping_method:
        new_col_name = self.variables + '_Transaction_' + self.groupping_method
        X[new_col_name] = X[self.groupby_variables].map(self.group_amount_dict_)
        return X[new_col_name].to_numpy().reshape(-1, 1)

    ############################################

class FeaturesEngineerCumCount(BaseEstimator, TransformerMixin):
    def __init__(self, group_one="step",
                 group_two="nameOrig"
                 ):
        self.group_one = group_one
        self.group_two = group_two

    def fit(self, X, y=None):
        """
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        new_col_name = self.group_two + '_Transaction_count'
        X[new_col_name] = X.groupby([self.group_one, self.group_two])[[self.group_two]].transform('count')
        return X[new_col_name].to_numpy().reshape(-1, 1)
