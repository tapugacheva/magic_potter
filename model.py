import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostRegressor
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from preproccess import *
from paths import *
import warnings

warnings.filterwarnings("ignore")


class Model:
    def __init__(self):
        self.df = read_merge_train(pricing_hackathon_checks_train, pricing_hackathon_shops, pricing_hackathon_hierarchy,
                                   pricing_hackathon_prices_train)
        self.model = CatBoostRegressor(iterations=200, loss_function='RMSE')
        self.dates = get_datelist('2021-06-30', '2018-01-01')
        self.df.fillna(0, inplace=True)

    def pred_full(self):
        print('read')
        self.df.set_index('day', inplace=True)
        self.df.sort_index(inplace=True)
        train, val = self.df.loc[:20210531], self.df.loc[20210601:]
        print('split')
        features_train = train.drop(['num_sales'], axis=1)
        target_train = train['num_sales']
        features_test = val.drop(['num_sales'], axis=1)
        target_test = val['num_sales']
        model = self.model
        model.fit(features_train, target_train, cat_features=self.cats())
        preds = model.predict(features_test)
        val['preds'] = preds
        val.to_csv('./preds.csv')
        print('WAPE', self.wape(target_test.values, preds))
        print('mape', mean_absolute_percentage_error(target_test.values, preds))
        shap_test = shap.TreeExplainer(model).shap_values(val)
        shap.summary_plot(shap_test, val,
                          max_display=25, auto_size_plot=True)
        total_test = read_merge_test(pricing_hackathon_checks_test, pricing_hackathon_shops,
                                     pricing_hackathon_hierarchy, pricing_hackathon_prices_test)
        test_preds = model.predict(total_test)
        total_test['preds'] = test_preds
        total_test.to_csv('./preds.csv')
        print('!!!')
        return test_preds

    @staticmethod
    def cats():
        return ['class', 'group', 'category', 'subcategory']

    @staticmethod
    def wape(y_true: np.array, y_pred: np.array):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


if __name__ == '__main__':
    cat = Model()
    cat.pred_full()
