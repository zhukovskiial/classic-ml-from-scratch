import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def woe_continuous(data, target, bins=10):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns

    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        # Calculate the number of events in each group (bin)
        d = d0.groupby("x", as_index=False, observed=True).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']

        # Calculate % of events in each group.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

        # Calculate the non events in each group.
        d['Non-Events'] = d['N'] - d['Events']

        # Calculate % of non events in each group.
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # Calculate WOE by taking natural log of division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
        
    #Show WOE Table    
    return woeDF


def woe_discrete(df, cat_variable_name, y_df):
    """
    Assumes the categorical variable is already in a binned/categorized form.
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a categorical variable.
    """
    # Combine the feature and target variable for easier manipulation
    data = pd.concat([df[[cat_variable_name]], y_df], axis=1)
    data.columns = ['feature', 'target']

    # Calculate the necessary statistics
    grouped = data.groupby('feature')['target'].agg(['count', 'mean'])
    grouped.columns = ['n_obs', 'prop_good']
    grouped['prop_n_obs'] = grouped['n_obs'] / grouped['n_obs'].sum()
    grouped['n_good'] = grouped['prop_good'] * grouped['n_obs']
    grouped['n_bad'] = grouped['n_obs'] - grouped['n_good']
    grouped['prop_n_good'] = grouped['n_good'] / grouped['n_good'].sum()
    grouped['prop_n_bad'] = grouped['n_bad'] / grouped['n_bad'].sum()

    # Calculate WoE
    grouped['WoE'] = np.log((grouped['prop_n_good'] + 0.0001) / (grouped['prop_n_bad'] + 0.0001))
    grouped['IV'] = (grouped['prop_n_good'] - grouped['prop_n_bad']) * grouped['WoE']
    total_IV = grouped['IV'].sum()

    # Return the WoE and IV values
    grouped['total_IV'] = total_IV
    return grouped[['WoE', 'IV', 'total_IV']]
