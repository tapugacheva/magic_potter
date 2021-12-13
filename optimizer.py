import pandas as pd
from scipy.optimize import basinhopping
import numpy as np
from tqdm import tqdm
from preproccess import *
import warnings

warnings.filterwarnings("ignore")


class Opt:
    def __init__(self):
        self.df = pd.read_csv('./preds.csv')


    def anneal(self, price):
        minimizer_kwargs = {"method": "TNC"}
        x0 = [price]
        func = lambda x: 0.9762 + 0.0603 * np.log(x)
        opt = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs)
        return opt


if __name__ == '__main__':
    opt = Opt()
    opt.df['opt_price'] = 0
    opt.df['opt_sales'] = 0
    for i in tqdm(range(400)):
        if opt.df['reg_price'].iloc[i] > 0:
            price = opt.df['reg_price'].iloc[i]
            ret = opt.anneal(price)
            opt.df['opt_price'].iloc[i] = ret.x
            opt.df['opt_sales'].iloc[i] = ret.fun
    opt.df.to_csv('./opt_preds.csv')
