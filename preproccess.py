import pandas as pd
from paths import *


def read_merge_train(data, shops, skus, prices):
    full_train = pd.read_csv(data, header=None)
    full_train.columns = ['client_id', 'day', 'shop_id', 'check_id', 'time', 'sku', 'promo_id', 'check_pos',
                          'num_sales', 'supplier_price', 'selling_price', 'discount', 'region_name']
    full_train['num_sales'] = full_train['check_pos'] * full_train['num_sales']
    shops_info = pd.read_csv(shops, header=None)
    shops_info.columns = ['division_id', 'region_id', 'city_id', 'shop_id', 'type_loc_id', 'type_size_id',
                          'type_format_id', 'type_wealth_id', 'is_store', 'is_active', 'is_ex_billa']
    skus_info = pd.read_csv(skus, header=None)
    skus_info.columns = ['sku', 'class', 'group', 'category', 'subcategory']
    prices_df = pd.read_csv(prices, header=None)
    prices_df.columns = ['day', 'shop_id', 'sku', 'promo_id', 'reg_price', 'promo_price', 'reg_card_price']
    prices_df['promo_id'].fillna(0, inplace=True)
    prices_df['promo_id'] = prices_df['promo_id'].apply(lambda x: 1 if x != 0 else x)
    prices_df.set_index(['day', 'shop_id', 'sku'], inplace=True)
    full_train = full_train.merge(shops_info, on='shop_id')
    full_train = full_train.merge(skus_info, on='sku')
    full_train = full_train.groupby(['day', 'shop_id', 'sku']).agg(
        {'num_sales': 'sum', 'supplier_price': 'mean', 'selling_price': 'mean', 'discount': 'mean',
         'type_loc_id': 'mean', 'type_size_id': 'mean',
         'type_format_id': 'mean', 'type_wealth_id': 'mean', 'is_store': 'mean', 'is_active': 'mean',
         'is_ex_billa': 'mean', 'class': 'first',
         'group': 'first', 'category': 'first', 'subcategory': 'first'})
    full_train = prices_df.join(full_train).reset_index()
    full_train = full_train[['day', 'supplier_price', 'selling_price', 'discount', 'type_loc_id', 'type_size_id',
                             'type_format_id', 'type_wealth_id', 'is_store', 'is_active', 'is_ex_billa', 'class',
                             'group', 'category', 'subcategory', 'promo_id', 'reg_price', 'promo_price',
                             'reg_card_price', 'num_sales']]
    full_train['year'] = full_train['day'].apply(lambda x: int(x // 1e4))
    full_train['month'] = full_train['day'].apply(lambda x: int(x % 1e4 // 1e2))
    full_train['dday'] = full_train['day'].apply(lambda x: int(x % 1e4 % 1e2))
    full_train.fillna(0, inplace=True)
    return full_train


def read_merge_test(data, shops, skus, prices):
    full_test = pd.read_csv(data, header=None)
    full_test.columns = ['client_id', 'day', 'shop_id', 'check_id', 'time', 'sku', 'promo_id', 'supplier_price',
                         'selling_price', 'discount', 'region_name']
    shops_info = pd.read_csv(shops, header=None)
    shops_info.columns = ['division_id', 'region_id', 'city_id', 'shop_id', 'type_loc_id', 'type_size_id',
                          'type_format_id', 'type_wealth_id', 'is_store', 'is_active', 'is_ex_billa']
    skus_info = pd.read_csv(skus, header=None)
    skus_info.columns = ['sku', 'class', 'group', 'category', 'subcategory']
    prices_df = pd.read_csv(prices, header=None)
    prices_df.columns = ['day', 'shop_id', 'sku', 'promo_id', 'reg_price', 'promo_price', 'reg_card_price']
    prices_df['promo_id'].fillna(0, inplace=True)
    prices_df['promo_id'] = prices_df['promo_id'].apply(lambda x: 1 if x != 0 else x)
    prices_df.set_index(['day', 'shop_id', 'sku'], inplace=True)
    full_test = full_test.merge(shops_info, on='shop_id')
    full_test = full_test.merge(skus_info, on='sku')
    full_test = full_test.groupby(['day', 'shop_id', 'sku']).agg(
        {'supplier_price': 'mean', 'selling_price': 'mean', 'discount': 'mean',
         'type_loc_id': 'mean', 'type_size_id': 'mean',
         'type_format_id': 'mean', 'type_wealth_id': 'mean', 'is_store': 'mean', 'is_active': 'mean',
         'is_ex_billa': 'mean', 'class': 'first',
         'group': 'first', 'category': 'first', 'subcategory': 'first'})
    full_test = prices_df.join(full_test).reset_index()
    full_test = full_test[['day', 'shop_id', 'sku', 'supplier_price', 'selling_price', 'discount', 'type_loc_id', 'type_size_id',
                             'type_format_id', 'type_wealth_id', 'is_store', 'is_active', 'is_ex_billa', 'class',
                             'group', 'category', 'subcategory', 'promo_id', 'reg_price', 'promo_price',
                             'reg_card_price', 'num_sales']]
    full_test['year'] = full_test['day'].apply(lambda x: int(x // 1e4))
    full_test['month'] = full_test['day'].apply(lambda x: int(x % 1e4 // 1e2))
    full_test['dday'] = full_test['day'].apply(lambda x: int(x % 1e4 % 1e2))
    full_test.fillna(0, inplace=True)
    full_test.set_index(['day', 'shop_id', 'sku'], inplace=True)
    return full_test


def get_datelist(end_date, start_date):
    n_days = (pd.to_datetime(end_date).date() - pd.to_datetime(start_date).date()).days
    datelist = pd.date_range(pd.to_datetime(start_date).date(), periods=n_days + 1).tolist()
    return datelist


def get_sales_for_opt(sales, prices):
    sales_df = pd.read_csv(sales, header=None)
    sales_df.columns = ['client_id', 'day', 'shop_id', 'check_id', 'time', 'sku', 'promo_id', 'check_pos',
                        'num_sales', 'supplier_price', 'selling_price', 'discount', 'region_name']
    prices_df = pd.read_csv(prices, header=None)
    prices_df.columns = ['day', 'shop_id', 'sku', 'promo_id', 'reg_price', 'promo_price', 'reg_card_price']
    sales_df = sales_df.groupby(['day', 'shop_id', 'sku']).agg({'num_sales': 'sum'})
    prices_df.set_index(['day', 'shop_id', 'sku'], inplace=True)
    return prices_df.join(sales_df)
