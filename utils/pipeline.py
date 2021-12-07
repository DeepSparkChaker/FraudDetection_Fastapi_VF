from utils.preparation import *
# complete pipe :
# select the float/cat columns
#cat_feautres = X.select_dtypes(include=['object','category']).columns
#num_features = X.select_dtypes(exclude=['object','category']).columns
#Define vcat pipeline
features_cum_count=['step','nameOrig']
features_groupby_amount=['amount','nameOrig']
features_frequency_orig_dest=['nameOrig','nameDest']
features_cum_count_pipe = Pipeline([
                     ('transformer_Encoder', FeaturesEngineerCumCount())
                    ])
features_groupby_pipe = Pipeline([
                     ('transformer_group_amount_mean', FeaturesEngineerGroup2()),
                     ('transformer_group_scaler', PowerTransformer())
                    ])
features_frequency_pipe = Pipeline([
                     ('Encoder', CountFrequencyEncoder(variables=['nameOrig','nameDest'],encoding_method ="frequency", keep_variable=False))
                    ])
type_pipe= Pipeline([
                     ('transformer_Encoder', ce.cat_boost.CatBoostEncoder())
                    ])
num_features0=[  'amount',  'oldbalanceOrig', 'newbalanceOrig' ,'oldbalanceDest', 'newbalanceDest']
#Define vnum pipeline
num_pipe = Pipeline([
                     ('scaler', PowerTransformer()),
                    ])
#Featureunion fitting training data
preprocessor = FeatureUnion(transformer_list=[('cum_count', features_cum_count_pipe),
                                              ('mean_amount', features_groupby_pipe),
                                              ('frequency_dest_orig', features_frequency_pipe),
                                              ('trans_type', type_pipe),
                                              ('num', num_pipe)])
data_preparing= ColumnTransformer([
    ('cum_count', features_cum_count_pipe, features_cum_count ),
    ('mean_amount', features_groupby_pipe, features_groupby_amount ),
    ('frequency_dest_orig', features_frequency_pipe, features_frequency_orig_dest ),
    ('trans_type', type_pipe, ['type'] ),
    ('num', num_pipe, num_features0)
], remainder='drop')