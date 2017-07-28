import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import time
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import izip, product


def dt_string_to_cerner_time(date_time_string):
    """
    Input: date_time_string (string)
    
    Output: unix_time_stamp (string)
    
    Description: Take a datetime string of the form "YYYY:MMM:DD:HH:MM" (ex: "2016:8:5:12:45") 
                 and convert it to a unix timestamp of the form that Cerner uses.
    """
    
    assert ':' in date_time_string, """ Make sure your date time is of the form 'YYYY:MMM:DD:HH' 
                                       (ex: '2016:8:5:12:45' is the 12:45pm on August 5th, 2016) """
    date_parts = [int(part) for part in date_time_string.split(':')]
    dt = datetime.datetime(*date_parts)
    unix_time = time.mktime(dt.timetuple())
    cerner_time = str(unix_time)[:-2] + '000'
    return cerner_time


def scatter_avgs(event_cd1, event_cd2, start_time_string, end_time_string):
    """
    Input: event_cds (string), start_time (string), end_time (string) 
    
    This function returns a figure, of the average of the vital signs per encounter
    
    Description: Returns averages of vitals data for event_cd1 and event_cd2 between 
                 start_time_string and end_time_string. Date time is of the form 'YYYY:MMM:DD:HH' 
                 (ex: '2016:8:5:12:45' is the 12:45pm on August 5th, 2016).
    """

    start_time = sdt_string_to_cerner_time(start_time_string)
    end_time = sdt_string_to_cerner_time(end_time_string)

    query_RRT = """
                SELECT 
                      ce.encntr_id 
                    , ce.event_cd 
                    , cv_event_cd.description AS event_description 
                    , AVG(CAST(ce.result_val as int)) as avg 
                FROM clinical_event ce  
                JOIN encounter enc ON ce.encntr_id = enc.encntr_id 
                LEFT OUTER JOIN code_value cv_event_cd 
                ON ce.event_cd = cv_event_cd.code_value 
                WHERE ce.encntr_id IN ( 
                                       SELECT DISTINCT encntr_id 
                                       FROM clinical_event 
                                       WHERE event_cd = '54411998' 
                                       AND result_status_cd NOT IN ('31', '36') 
                                       AND valid_until_dt_tm > unix_timestamp() 
                                       AND event_class_cd not in ('654645') 
                                      ) 
                AND ce.event_cd IN ('{0}', '{1}')
                AND ce.performed_dt_tm > {2} 
                AND ce.performed_dt_tm < {3} 
                AND enc.loc_facility_cd = '633867'
                GROUP BY ce.encntr_id, ce.event_cd, cv_event_cd.description 
                ORDER BY ce.encntr_id;
                """.format(str(event_cd1), str(event_cd2), start_time, end_time)
    
    cur.execute(query_RRT)
    df_RRT = as_pandas(cur)
    df_1RRT = df_RRT[df_RRT.event_cd==str(event_cd1)]
    df_2RRT = df_RRT[df_RRT.event_cd==str(event_cd2)]
    df_finRRT = pd.merge(df_1RRT, df_2RRT, on='encntr_id')
    
    query_NotRRT = """
                SELECT 
                      ce.encntr_id 
                    , ce.event_cd 
                    , cv_event_cd.description AS event_description 
                    , AVG(CAST(ce.result_val as int)) as avg 
                FROM clinical_event ce  
                JOIN encounter enc ON ce.encntr_id = enc.encntr_id 
                LEFT OUTER JOIN code_value cv_event_cd 
                ON ce.event_cd = cv_event_cd.code_value 
                WHERE ce.encntr_id NOT IN ( 
                                           SELECT DISTINCT encntr_id 
                                           FROM clinical_event 
                                           WHERE event_cd = '54411998' 
                                           AND result_status_cd NOT IN ('31', '36') 
                                           AND valid_until_dt_tm > unix_timestamp() 
                                           AND event_class_cd not in ('654645') 
                                           ) 
                AND ce.event_cd IN ('{0}', '{1}')
                AND ce.performed_dt_tm > {2} 
                AND ce.performed_dt_tm < {3}
                AND enc.loc_facility_cd = '633867'
                GROUP BY ce.encntr_id, ce.event_cd, cv_event_cd.description 
                ORDER BY ce.encntr_id;
                """.format(str(event_cd1), str(event_cd2), start_time, end_time))
    
    cur.execute(query_NotRRT)
    df_NotRRT = as_pandas(cur)
    df_1NotRRT = df_NotRRT[df_NotRRT.event_cd==str(event_cd1)]
    df_2NotRRT = df_NotRRT[df_NotRRT.event_cd==str(event_cd2)]
    df_finNotRRT = pd.merge(df_1NotRRT, df_2NotRRT, on='encntr_id')
    
    # plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(df_finNotRRT.avg_x, df_finNotRRT.avg_y, s = 40, alpha=0.5, c='blue')
    plt.scatter(df_finRRT.avg_x, df_finRRT.avg_y, s = 30, alpha=0.5, c='red')
    plt.xlabel(event_cd1)
    plt.ylabel(event_cd2)
    plt.legend(['Non-RRT patients', 'RRT patients'])
    plt.title('Vital sign averages per encounter')

def show_df_nans(df, columns=None, plot_width=10, plot_height=8):
    """
    Input: df (pandas dataframe), collist (list)
    
    Output: seaborn heatmap plot
    
    Description: Create a data frame for features which may be nan. Set NaN values be 1 and 
                 numeric values to 0. Plots a heat map where dark squares/lines show where 
                 data is missing. The columns to plot can be specified with an input param. Otherwise
                 all columns will be plotted -- which appear crowded.
    """

    if not columns:
        plot_cols = df.columns
    else:
        plot_cols = columns
    df_viznan = pd.DataFrame(data=1, index=df.index, columns=plot_cols)
    df_viznan[~pd.isnull(df[plot_cols])] = 0
    plt.figure(figsize=(plot_width, plot_height))
    plt.title('Dark values are nans')
    return sns.heatmap(df_viznan.astype(float))


def impute_with_linear_regression(df, response_col, explanatory_cols):
    """
    Input: df (pandas dataframe), response_col (col name as string), explanatory_cols (list of col names as strings)
    
    Output: seaborn heatmap plot
    
    Description: Fills missing values of response_col by using explanatory_cols to train 
                 a linear regression model to predict missing values of response_col.
                 * DOES NOT NORMALIZE DATA OR DO ANYTHING TO HANDLE COLLNEARITY *
    
    Usage: impute_with_linear_regression(df, response='heart_rate', explanatory_cols=['age', 'respiration_rate']) 
    """

    reg = LinearRegression()
    # Train a linear regression model on the rows where there are complete data
    train_index = df[explanatory_vars + [column]].dropna().index
    X_train = df[explanatory_vars].ix[train_index]
    y_train = df[column].ix[train_index]
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    print "R-squared of imputation model: ", r2_score(y_train, y_pred)
    # Use that model to predict the missing values of col
    imputation_index = [el for el in set(df.index) - set(X_train.index)]
    # Check to make sure our explanatory vars for the imputation model are not null
    assert df[explanatory_vars].ix[imputation_index].isnull().sum().sum() == 0, "One or more columns in explanatory_vars contains a null value"
    vals = reg.predict(df[explanatory_vars].ix[imputation_index])
    # Match the indices of the missing values with the right predicted value
    imputation_dict = {ix: val for ix, val in izip(imputation_index, vals)}
    df[col] = df[col].fillna(imputation_dict)
    return df


def fill_missing(df, column, method="mean", explanatory_vars=None):
    """
    Input: df (pandas dataframe), column (string), method (string), explanatory_vars (list of col names as strings)
    
    Output: df (pandas dataframe)
    
    Description: Imputes the missing values using one of three specified methods: "mean" (column mean), 
                "meadian" column median, or "regression" (a linear regression model to predict the 
                 missing values based on explanatory_cols).
    """

    if method == "mean":
        val = df[col][df[col].notnull()].mean()
    elif method == "median":
        val = df[col][df[col].notnull()].median()
    elif method == "regression":
        assert explanatory_vars, "You need to supply explanatory variables to create a regression model"
        imputed = impute_with_linear_regression(df, col, explanatory_vars)
        df[col] = imputed[col]
        return df
    else: assert False, "Select one of the following methods of imputation: 'mean', 'median', or 'regression'"
    df[col] = df[col].fillna(val)
    return df


def score_printout(X_test, y_test, fittedModel):
    """
    Input: X_test (pandas dataframe or numpy array), y_test (pandas dataframe or numpy array), 
           fittedModel (trained model object)
    
    Output: None (prints the AUROC, precision and recall scores)
    
    Description: A helper to quickly show some useful accuracy metrics for model evaluation.
    """

    print "AUC-ROC Score of model: ", roc_auc_score(y_test, fittedModel.predict_proba(X_test)[:,1])
    print "Precision Score of model: ", precision_score(y_test, fittedModel.predict(X_test))
    print "Recall Score of model: ", recall_score(y_test, fittedModel.predict(X_test))


def plot_roc_curve(X_test, y_test, fig_size=(10, 8)):
    """
    Inputs: 'X_test': (pandas dataframe or np array) The test features.
            
            'y_test': (pandas series or np array) The test labels.
            
            'fig_size': (tuple of ints) The size of the output plot.

    Output: None. Print a matplotlib plot of the the roc curve.

    Description: This function prints and plots the roc_auc curve.
    
    Usage: plot_roc_curve(X_test, y_test)
    """
    
    y_pred_proba = gbc.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=fig_size)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(fittedModel, X_test, y_test, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          fig_size=(8,8)):
    """
    Input:  'fittedModel': (model obj with a predict method) Example. A fitted GradientBoostedClassifier model.
    		
    		'X_test': (np array or pandas dataframe) The test data features.
    		
    		'y_test': (np array or pandas series) The test true labels.
    		
    		'classes': (list) List of the class names, e.g. classes=['noRRT', 'RRT']
            
            'normalize': (boolean) If True will express confusion matrix in proportions of the total true classes. Will show
 						 nominal values of otherwise. Default value is False.
            
            'title': (string) The title of the plot. Defaults to 'Confusion Matrix'.
            
            'cmap':  (string) The matplotlib color map attribute you want. Defaults to plt.cm.Blues
            
            'fig_size': (tuple of integers) The deimensions of the plot. Defaults to (8,8)

    Output: A matplotlib plot of the confusion matrix

    Description: This function prints and plots the confusion matrix.

    Usage: plot_confusion_matrix(X_test, y_test, normalize=True, classes=['class1', 'class2'])
    
    """
    
    cm = confusion_matrix(y_test, fittedModel.predict(X_test))
    plt.figure(figsize=fig_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=20)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_feature_importance_plot(fittedModel, X_train, numFeatures):
    """
    Input:  'fittedModel': (model obj with a predict method). Example. A fitted GradientBoostedClassifier model.
            'X_train': (array or dataframe). The training data that were used to build the model.
            'numFeatures': (int). The number of features that you want to plot.
    
    Output: Matplotlib bar plot
    
    Description: A helper to quickly plot the most important features from a model that has the 
                 feature_importance method, e.g. RandomForests or GradientBoostedMachines.
                 Ex: featuresAndImportances = trainedModel.feature_importances_
    """
    featuresAndImportances = izip(X_train.columns, fittedModel.feature_importances_)
    topN = sorted(featuresAndImportances, reverse=True, key=lambda x: x[1])[:numFeatures]
    labels = [pair[0] for pair in topN]
    values = [pair[1] for pair in topN]
    ind = np.arange(len(values))
    width = 0.35   
    plt.bar(range(numFeatures), values, width=0.8)
    ax = plt.subplot(111)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(labels, rotation=60, size=12)
    plt.xlabel('Feature', size=20)
    plt.ylabel('Importance', size=20)
    plt.show()