import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def woe_continuous(data, target, bins=10):
    
    """
    Функция преобразовывает числовые значения признака в значение weight of evidence для последующего использования в обучении линейной модели
    В функцию нужно передать датафрейм с двумя колонками: сам признак и таргет, а также название колонки с таргетом
    Данная реализация предполагает, что таргет это класс 0 или 1
    """
    
    woeDF = pd.DataFrame()
    cols = data.columns

    # разбиваем все значения на заданное число групп
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        # находим кол-во классов 1 в каждой группе
        d = d0.groupby("x", as_index=False, observed=True).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']

        # находим % классов 1 в каждой группе
        # 0.5 берется как замена для 0, потому что потом от этих значений находим логарифм
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

        # находим кол-во классов 0 в каждой группе
        d['Non-Events'] = d['N'] - d['Events']

        # находим % классов 0 в каждой группе
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # находим значение woe как логарифм от деления % класса 1 на % класса 0
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        # рассчитаем значение Information Value
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        woeDF=pd.concat([woeDF,d], axis=0)
            
    return woeDF


def woe_discrete(df, cat_variable_name, y_df):
    
    """
    Функция преобразовывает категориальные значения признака в значение weight of evidence для последующего использования в обучении линейной модели
    В функцию нужно передать датафрейм с двумя колонками: сам признак и таргет
    Данная реализация предполагает, что таргет это класс 0 или 1
    """
    
    data = pd.concat([df[[cat_variable_name]], y_df], axis=1)
    data.columns = ['feature', 'target']

    grouped = data.groupby('feature')['target'].agg(['count', 'mean'])
    grouped.columns = ['n_obs', 'prop_good']
    grouped['prop_n_obs'] = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good'] = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad'] = grouped['n_obs'] - grouped['n_good']
    grouped['prop_n_good'] = grouped['n_good'] / grouped['n_good'].sum()
    grouped['prop_n_bad'] = grouped['n_bad'] / grouped['n_bad'].sum()

    # прибавляем малое значение 0.0001, чтобы избежать случаев, когда передаем 0 в логарифм
    grouped['WoE'] = np.log((grouped['prop_n_good'] + 0.0001) / (grouped['prop_n_bad'] + 0.0001))
    grouped['IV'] = (grouped['prop_n_good'] - grouped['prop_n_bad']) * grouped['WoE']
    total_IV = grouped['IV'].sum()

    grouped['total_IV'] = total_IV
    return grouped[['WoE', 'IV', 'total_IV']]
