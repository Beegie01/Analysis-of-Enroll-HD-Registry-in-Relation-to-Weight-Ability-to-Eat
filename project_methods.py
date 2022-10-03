from seaborn.utils import pd, os, plt, np
import seaborn as sns
import pickle
import string
from itertools import combinations
import copy
import seaborn as sns
from matplotlib.transforms import Affine2D
from seaborn.utils import os, np, plt, pd
from sklearn import preprocessing as s_prep, metrics as s_mtr, cluster, decomposition as s_dec, feature_selection as s_fs, pipeline as s_pipe
from sklearn import utils as s_utils
from tensorflow.keras import models, layers, backend as K, callbacks, metrics as k_mtr, optimizers
from scipy import stats
from scipy.stats import ttest_ind_from_stats, ttest_ind
import statsmodels as stat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlxtend import frequent_patterns


class MSc_Proj:
    
    hd_status_guide = {2:'2. pre-manifest', 3:'3. manifest', 4:'4. genotype -ve', 5:'5. family controls'}
    variables = {'miscore':'incomplete_motor_score', 'tfcscore':'functional_score', 'fiscore':'incomplete_functional_score', 'indepscl':'percentage_independence',
                'grocery':'independent_shopping', 'prepmeal':'independent_cooking', 'ownmeds':'independent_drug_ingestion', 'feedself':'independent_feeding',
                'walkhelp':'independent_walking', 'carehome':'care_provided_at_home', 'depscore':'depression', 'aptscore':'apathy', 'exfscore':'executive_function',
                'wpaiscr4':'activity_impaired_by_hd', 'mvrsn':'reason_for_missed_visit', 'crlvl':'required_level_of_care', 'caghigh':'larger_research_cag_allele',
                'caglow':'smaller_research_cag_allele', 'momhd':'mother_of_participant_had_hd', 'dadhd':'father_of_participant_had_hd', 'ccmtr':'participant_has medical_history_of_hd_motor_symptoms',
                'ccdep':'participant_has_medical_history_of_depression', 'ccapt':'participant_has_medical_history_of_apathy', 'cccog':'participant_has_medical_history_of_significant_cognitive_impairment', 
                'sxgs':'best_guess_of_how_many_years_ago_symptoms_onset_occurred', 'cmcat':'type_of_nutritional_supplement'}
    dose_freq_guide = {1:'daily', 2:'every other day', 3:'every third day', 4:'weekly', 5:'every other week',
                       6:'monthly', 7:'every other month', 8:'quarterly', 9:'annually', 10:'as needed'}
    drug_intake_guide = {'p.o':1, 'p.r':2, 's.c':3, 'i.m':4, 'i.v':5, 'nasal':6, 'td':7, 'sl':8, 'inh':9, 'other':10}
    
    bmi_change_guide = {-999:'first_visit', 0:'no_change', -1:'decrease', 1:'increase'}
    bmi_change_cmap = {'no_change':'gray', 'decrease': 'red', 'increase':'green', 'first_visit':'blue'}
    bmi_change_int_cmap = {0:'gray', -1: 'red', 1:'green', -999:'blue'}
    bmi_lvl_guide = {'0. underweight':0, '1. normal':1, '2. overweight':2, '3. obese':3, '4. severely obese':4}
    bmi_level_cmap = {'0. underweight':'yellow', '1. normal':'green', '2. overweight':'gray', '3. obese':'red', '4. severely obese':'blue'}
    bmi_outcome_guide = {-1:'underweight', 0:'normal', 1:'overweight'}
    bmi_outcome_int_cmap = {-1:'red', 0:'green', 1:'black'}
    bmi_outcome_cmap = {'underweight':'red', 'normal':'green', 'overweight':'black'}
    yn_guide = {0:'no', 1:'yes'}
    yn_cmap = {'yes':'darkblue', 'no':'gray'}
    age_band = {'five_yr_band': {1:'below 18', 2:'18-24', 3:'25-29', 4:'30-34', 5:'35-39', 6:'40-44', 7:'45-49', 8:'50-54', 9:'55-59', 10:'60-64', 11:'65-69', 12:'above 70'},
                'ten_yr_band': ('0. <30', '1. 30 - 39', '2. 40 - 49', '3. 50 - 59', '4. 60 - 69', '5. >=70')}
    gender_cmap = {'f':'blue', 'm':'brown'}
    hd_status_cmap = {'2. pre-manifest':'yellow', '3. manifest':'black', '5. family controls':'brown', '4. genotype -ve':'gray'}
    cagrepeat_band_cmap = {'0. normal':'green', '1. intermediate':'yellow', '2. reduced penetrance':'gray', '3. full penetrance':'red'}
    
    
    def __init__(self):
        os.makedirs(self.app_folder_loc, exist_ok=True)  # create folder 'Files' at current working directory for storing useful files
        
    @property
    def app_name(self):
        return 'MScProject'
        
    @property
    def app_folder_loc(self):
        return f'{self.app_name}Output'

    @staticmethod
    def save_python_obj(fname, py_obj):
        """save python object to file system.
        NOTE: always add .pkl to fname"""
        
        with open(fname, 'wb') as f:
            pickle.dump(py_obj, f)
        print("Python object has been saved")
        
    @staticmethod
    def load_python_obj(fname):
        """load python object from filesystem
        NOTE: always include filename extension"""
        
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        print('Loading complete')
        return obj
        
    @staticmethod
    def get_features_with_dtypes(df, feat_datatype: str='number'):
        """get a list of features with specified datatype in the dataframe.
        default output is numeric features.
        feat_datatype: options are number/int/float, object/str, bool, datetime/date/time
        Return: feat_list"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        if not isinstance(feat_datatype, str):
            feat_datatype = eval(str(feat_datatype).split()[1][:-1])
        
        guide = {'str': 'object', 'int': 'number', 'float': 'number', 'date': 'datetime', 'time': 'datetime'}
        if feat_datatype in guide:
            use_datatype = guide[feat_datatype]
            #print(use_datatype)
            cols_selected = df.select_dtypes(include=use_datatype)
        else:
            cols_selected = df.select_dtypes(include=feat_datatype)
        
        if feat_datatype in ['int', 'float']:
            
            print(feat_datatype)
            col_dtypes = cols_selected.dtypes.astype(str).str.lower()
            return list(col_dtypes[col_dtypes.str.contains(feat_datatype)].index)
        #print('yes')
        return list(cols_selected.columns)
    
    @staticmethod
    def upsample_imbalanced_data(X: pd.DataFrame, y: pd.Series):
            """oversample minority class instances.
            Return:
            X_upsampled, y_upsampled."""
            
            from imblearn import over_sampling
            cls = MSc_Proj()
            
            catg_cols = cls.get_features_with_dtypes(X, str)
            num_cols = cls.get_features_with_dtypes(X)
            
            if len(catg_cols):  # presence of categorical features
                # get column indexes for categorical columns
                col_ind = {v:i for i, v in list(enumerate(X.columns.astype(str)))}

                catg_ind = [col_ind[col] for col in catg_cols]
                if len(catg_ind) == 1:
                    catg_ind = catg_ind[0]
                #print(catg_ind)
            
                if len(num_cols):
                    # numeric and categorical upsampler
                    over_sampler = over_sampling.SMOTENC(categorical_features=(catg_ind,), random_state=1)
                else:
                    over_sampler = over_sampling.SMOTEN(random_state=1)
            else:
                over_sampler = over_sampling.SMOTE(random_state=1)
                
            return over_sampler.fit_resample(X, y)
            
    @staticmethod
    def get_coeff_importance(trained_model: 'dataframe', plot_title=None):
        """determine feature importance from coefficient values (beta) of predictors
        based on contribution to model prediction.
        trained_model: instance of trained model having coefficients
        Return
        feature_importance: series ranking of features"""
        
        cls = MSc_Proj()
        
        weights = pd.DataFrame(trained_model.coef_, columns=trained_model.feature_names_in_)
        weights.loc[:, 'intercept'] = trained_model.intercept_
        print('\nModel Coefficients per Target Outcome:')
        display(weights)

        feat_importance = weights.drop('intercept', axis=1).apply(lambda row: np.e**row).round(decimals=2).mean().sort_values(ascending=False)
    #     display(feat_importance)
        if not plot_title:
            plot_title = 'Feature Contribution to Model Prediction'
        cls.plot_column(feat_importance.index, feat_importance.round(2),
                        plot_title=plot_title,
                        rotate_xticklabe=90, color='brown', reduce_barw_by=1.5)
        return feat_importance

    @staticmethod
    def reduce_train_test_dims(xtrain, xtest, pca_pct=0.95):
        """reduced an already split and rescaled dataset train and test dataset
        pca_pct: percentage of explained variance for pca
        Return
        reduced_xtrain, reduced_xtest"""
        
        cls = MSc_Proj()
        # engineer a row indicator to distinguish between train and test data
        xtrain.loc[:, 'train_set'] = 1
        xtest.loc[:, 'train_set'] = 0

        # join train set to test set before PCA
        train_test_rows = pd.concat([xtrain, xtest], ignore_index=True)
        # record test data index
        test_ix = train_test_rows.loc[train_test_rows['train_set'] == 0].index
        # reduce train_test data
        xreduced, pca_comps = cls.train_PCA(train_test_rows, perc_components=pca_pct)
        # select train and test set data rows
        reduced_xtrain = xreduced.loc[~xreduced.index.isin(test_ix)].reset_index(drop=True)
        reduced_xtest = xreduced.loc[test_ix].reset_index(drop=True)
        
        return reduced_xtrain, reduced_xtest, pca_comps

    @staticmethod
    def train_PCA(df, perc_components=0.95):
        """Reduce the dimension a dataframe using a PCA model.
        principal components explain a proportion of the variance
        of the given input
        perc_components: fraction of the df variance you want explained 
        by the output
        
        NOTE: Rescale the input to between (0, 1) BEFORE 
        using PCA
        
        Return 
        (compressed_df, PCA_comps)"""
        
        if not isinstance(perc_components, float):
            raise ValueError('please input the percentage of the input variance you want explained by the components')
        
        if (perc_components > 1.0) or (perc_components < 0):
            raise ValueError('please input a percentage of within range (0, 1)')
        
        np.random.seed(1)
        pca = s_dec.PCA(perc_components, random_state=1)
        df_pca = pd.DataFrame(np.round(pca.fit_transform(df), 2))
        cc = {i: f'Component_{i+1}' for i in range(len(pca.components_))}
        df_pca.columns = cc.values()
        
        pca_comp = pd.DataFrame(np.round(pca.components_, 2), columns=pca.feature_names_in_)
        expl_var =  pd.DataFrame({'explained_variance_perc': np.round(pca.explained_variance_ratio_, 2)})
        pca_comp = pd.concat([pca_comp, expl_var], axis=1)
        
        print(f"These {len(pca.components_)} components explains {np.round(100*pca_comp['explained_variance_perc'].sum(), 2)}% of input variance\n")
        print(f'The variance of each component is given as:\n{100*np.cumsum(pca.explained_variance_ratio_).round(2)}')
        
        return df_pca, pca_comp
    @staticmethod
    def check_for_pvalue(stat_df, col_name:str):
    
        cls = MSc_Proj()
        unq_combo = cls.get_unique_combinations(stat_df.columns, 2)
        display(unq_combo)
        h1 = 'two-sided'
        print(str.upper(h1))
        for a, b in unq_combo:
            print(a, b)
            display(cls.get_ttest_pvalue_from_stats(X1_mean=stat_df[a].loc[(col_name, 'mean')], X1_std=stat_df[a].loc[(col_name, 'std')], X1_count=stat_df[a].loc[(col_name, 'count')],
                                                    X2_mean=stat_df[b].loc[(col_name, 'mean')], X2_std=stat_df[b].loc[(col_name, 'std')], X2_count=stat_df[b].loc[(col_name, 'count')],
                                                    alternative_hypothesis=h1))
            print('\n\n')

        h1 = 'less'
        print(str.upper(h1))
        for a, b in unq_combo:
            print(a, b)
            display(cls.get_ttest_pvalue_from_stats(X1_mean=stat_df[a].loc[(col_name, 'mean')], X1_std=stat_df[a].loc[(col_name, 'std')], X1_count=stat_df[a].loc[(col_name, 'count')],
                                                    X2_mean=stat_df[b].loc[(col_name, 'mean')], X2_std=stat_df[b].loc[(col_name, 'std')], X2_count=stat_df[b].loc[(col_name, 'count')],
                                                    alternative_hypothesis=h1))
            print('\n\n')

        h1 = 'greater'
        print(str.upper(h1))
        for a, b in unq_combo:
            print(a, b)
            display(cls.get_ttest_pvalue_from_stats(X1_mean=stat_df[a].loc[(col_name, 'mean')], X1_std=stat_df[a].loc[(col_name, 'std')], X1_count=stat_df[a].loc[(col_name, 'count')],
                                                    X2_mean=stat_df[b].loc[(col_name, 'mean')], X2_std=stat_df[b].loc[(col_name, 'std')], X2_count=stat_df[b].loc[(col_name, 'count')],
                                                    alternative_hypothesis=h1))
            print('\n\n')
    
    
    @staticmethod
    def get_unique_combinations(unq_elements: 'array of unique elements', n_combo_elements):
        """create combination of unique elements where each combination is n_combo_elements"""
        
        return [c  for c in combinations(list(unq_elements), n_combo_elements)]

    @staticmethod
    def get_correlcoeff_with_pvalues(X, y, show_pval_thresh=1):
        """compute pearson's correlation coefficient (and pvalues) between
        X variables and y
        Return
        result_df: if y is 1D, dataframe containing pearson coefficient (r) and pvalues
        else, dictionary containing y categories as keys and their r and pvalues for each
        variable"""
        
        
        def get_pearsonr_pval(X, y):
            """compute pearson's correlation coefficient (and pvalues) between
            X variables and y
            Return
            result_df: dataframe containing pearson coefficient (r) and pvalues"""
            r_dict, pval_dict = dict(), dict()

            X = pd.DataFrame(X)
            cols = X.columns

            for x in cols:
                r, pval = stats.pearsonr(X[x], y)
                r_dict[f'{x}'] = r
                pval_dict[f'{x}'] = pval
            return pd.DataFrame([pd.Series(r_dict,name='r'),
                                   pd.Series(pval_dict, name='pval')]).T.sort_values(by='pval')
        
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            y_dum = pd.get_dummies(pd.DataFrame(y))
        
            y_catg = dict()
            for y_col in y_dum:
                y = y_dum[y_col]
                result = get_pearsonr_pval(X, y)
                y_catg[f'{y_col}'] = result.loc[result['pval'] < show_pval_thresh]
            
            return y_catg
        
        result = get_pearsonr_pval(X, y)
        return result.loc[result['pval'] < show_pval_thresh]
                               
    @staticmethod
    def compute_median_std_sterr(arr, group_name='group', precision=2):
        """compute median, standard deviation and standard error for arr
        Return:
        result_df: dataframe reporting median, std_from_median, and sterror"""
        
        grp_mdn = pd.Series(np.median(arr).round(precision), name=f'{group_name}_median')
        
        # standard deviation from median
        dev = arr - arr.median()
        sq_dev = dev**2
        sumsq_dev = sq_dev.sum()
        deg_freedom = len(arr) -1
        variance = sumsq_dev/deg_freedom
        mdn_std = variance**0.5
        grp_std = pd.Series(np.round(mdn_std, precision), name=f'{group_name}_std')
        
        # standard error from median
        mdn_sterr = np.round(mdn_std/len(arr)**0.5, precision)
        grp_sterr = pd.Series(mdn_sterr, name=f'{group_name}_sterr')

        return pd.concat([grp_mdn, grp_std, grp_sterr], axis=1)

    @staticmethod
    def compute_mean_std_sterr(arr, group_name='group', precision=2):
        """compute mean, standard deviation and standard error for arr
        Return:
        result_df: dataframe reporting mean, std, and sterror"""
        
        grp_avg = pd.Series(np.mean(arr).round(precision), name=f'{group_name}_avg')
        grp_std = pd.Series(np.std(arr, ddof=1).round(precision), name=f'{group_name}_std')
        grp_sterr = pd.Series(np.round(np.std(arr, ddof=1)/np.sqrt(len(arr)), precision), 
                              name=f'{group_name}_sterr')
        
        return pd.concat([grp_avg, grp_std, grp_sterr], axis=1)
        
    @staticmethod
    def compute_balanced_weights(y: 'array', as_samp_weights=True):
        """compute balanced sample weights for unbalanced classes.
        idea is from sklearn:
        balanced_weight = total_samples / (no of classes * count_per_class)
        WHERE:
        total_samples = len(y)
        no of classes = len(np.unique(y))
        no of samples per class = pd.Series(y).value_counts().sort_index().values
        unique_weights = no of classes * no of samples per class
        samp_weights = {labe:weight for labe, weight in zip(np.unique(y), unique_weights)}
        
        weight_per_labe = np.vectorize(lambda l, weight_dict: weight_dict[l])(y, samp_weights)
        Return:
        weight_per_labe: if samp_weights is True
        class_weights: if samp_weights is False"""
        
        y = np.array(y)
        n_samples = len(y)
        n_classes = len(np.unique(y))
        samples_per_class = pd.Series(y).value_counts().sort_index(ascending=True).values
        denom = samples_per_class * n_classes
        unique_weights = n_samples/denom
        cls_weights = {l:w for l, w in zip(np.unique(y), unique_weights)}
        
        if as_samp_weights:
            return np.vectorize(lambda l, weight_dict: weight_dict[l])(y, cls_weights)
        return cls_weights
    
    @staticmethod
    def run_glm_predictor(X, y, binary_class=True, rescale=False, precision=2):
        """run a generalised linear model (logisitic regression) on the relationship between
        the dataframe's x_cols and y_col
        Return 
        model_summary: glm model summary object"""
        
        def generate_formula(X, y):
            """generate formula in string format"""
            cols, formula = X.columns, ""
            for n in range(len(cols)):
                if n == 0:
                    formula += cols[n]
                    continue
                formula += f"+{cols[n]}"
            return f"{y.name} ~ {formula}"
        
        if rescale:
            # normalise X
            scaler = s_prep.MinMaxScaler()
            scx = scaler.fit_transform(X)
            scX = pd.DataFrame(scx, columns=X.columns)
        else:
            scX = X
        Xc = sm.add_constant(scX)
        
        # create dataframe comprising only normalised X and y
        rlike_df = pd.merge(Xc, y, left_index=True, right_index=True)
        
        # create formula string consisting of dependent variable and independent variables
        formula = generate_formula(scX, y)
        
        # run glm model
        if binary_class:
            model = smf.glm(formula=formula, data=rlike_df, family=sm.families.Binomial())
            tmodel = model.fit()
        else:
#             model = smf.mnlogit(formula=formula, data=rlike_df, optimizer='powell')#, family=sm.families.Poisson())
            model = smf.glm(formula=formula, data=rlike_df)
#             model = stat.discrete.discrete_model.MultinomialModel(endog=y, exog=Xc)
#             model = sm.MNLogit(endog=y, exog=Xc)
            if rescale:
                tmodel = model.fit_regularized()
            else:
                tmodel = model.fit_regularized()
        
        print(f'Dependent Variable:\n{model.endog_names}, \n\nPValues:\n{tmodel.pvalues}, \n\nCoefficients:\n{tmodel.params}')
        return tmodel
        
    @staticmethod
    def dataset_split(x_array: 'np.ndarray', y_array: 'np.ndarray', perc_test=0.25, perc_val=None):
        """create train, validation, test sets from x and y dataframes
        Returns:
        if  val_len != 0:  # perc_val is not 0
            (x_train, x_val, x_test, y_train, y_val, y_test)
        else:  # perc_val is 0 (no validation set)
            (x_train, x_test, y_train, y_test)"""

        if not (isinstance(x_array, np.ndarray) or not isinstance(y_array, np.ndarray)):
            raise TypeError("x_array/y_array is not np.ndarray")

        nx, ny = len(x_array), len(y_array)

        if nx != ny:
            raise ValueError("x_array and y_array have unequal number of samples")
            
        if perc_val is None:
            perc_val = 0

        # number of samples for each set
        combo_len = int(nx * perc_test)
        train_len = int(nx - combo_len)
        val_len = int(combo_len * perc_val)
        test_len = combo_len - val_len

        print(f"Training: {train_len} samples")
        print(f"Validation: {val_len} samples")
        print(f"Test: {test_len} samples")

        if sum([train_len, test_len, val_len]) != nx:
            print("Error in computation")
            return None

        # indexes for x_array/y_array
        inds = np.arange(nx)
        np.random.shuffle(inds)

        # random indexes of test, val and training sets
        train_ind = inds[: train_len]
        combo_ind = inds[train_len:]
        
        test_ind = combo_ind[: test_len]
        val_ind = combo_ind[test_len: ]
            
        x_test, y_test = x_array[test_ind], y_array[test_ind]
        x_val, y_val = x_array[val_ind], y_array[val_ind]
        x_train, y_train = x_array[train_ind], y_array[train_ind]

        return (x_train, x_val, x_test, y_train, y_val, y_test) if val_len else (x_train, x_test, y_train, y_test)
        
    @staticmethod
    def report_with_conf_matrix(y_true, pred):
        """print classification report and return the corresponding
        confusion matrix.
        Return: confusion_matrix_plot"""
        
        print(s_mtr.classification_report(y_true, pred))
        
        sns.set_style('white')
        ax1 = s_mtr.ConfusionMatrixDisplay.from_predictions(y_true, pred)
        plt.title("Confusion Matrix", weight='bold')
        return ax1
        
    @staticmethod
    def get_correl_with_threshold(df: pd.DataFrame, thresh: float=0.5, report_only_colnames=False):
        """select only variables with correlation equal to or above the threshold value.
        Return:
        df_corr"""
        
        def count_thresh_corrs(row, thresh):
            """to count how many values in each row >= thresh"""
            
            count = 0
            for val in row:
                if abs(val) >= abs(thresh):
                    count += 1
            return count
        
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        print(f'Features with correlation of {round(100*thresh, 2)}% and above:\n')
        # number of values >= threshold value per row
        all_corrs = df.corr().round(4)
        selected_corr = all_corrs.apply(lambda val: count_thresh_corrs(val, thresh))
        correlated_vars = selected_corr[selected_corr > 1]
        if report_only_colnames:
            return list(correlated_vars.index)
        return all_corrs.loc[all_corrs.index.isin(list(correlated_vars.index)), list(correlated_vars.index)]
        
    @staticmethod
    def get_columns_with_pattern(df, search_pattern:'str or list', find_exact_match=False):
        """select from dataframe all columns containing the search pattern(s)."""
        
        if isinstance(search_pattern, str): # when search pattern is str
            if not find_exact_match:
                cols = [c for c in df.columns if str.lower(search_pattern) in c.lower()]
            else:
                cols = [c for c in df.columns if str.lower(search_pattern) == c.lower()]
        elif isinstance(search_pattern, list): # when list of search pattern is given
            cols = list()
            for c in df.columns:
                for search_for in search_pattern:
                    if not find_exact_match and str(search_for).lower() in c.lower():
                        cols.append(c)
                    elif find_exact_match and (str(search_for).lower() == c.lower()):
                        cols.append(c)
        return df[cols]
        
    @staticmethod
    def get_baseline_last_vis(df):
        """include only the baseline and last variables"""
        
        cls = MSc_Proj()
        vis_cols = cls.get_columns_with_pattern(df, 'vis_').columns
        v = [int(c.split('vis_')[-1]) for c in vis_cols]
        c = [c.split('vis_')[0] for c in vis_cols]
        min_vis, max_vis = f"vis_{min(v)}", f"vis_{max(v)}"
        vis_b = [f"{col}{min_vis}" for col in np.unique(c)]
        vis_l = [f"{col}{max_vis}" for col in np.unique(c)]
        vis_bl = list()
        for b, l in  zip(vis_b, vis_l):
            vis_bl.append(b)
            vis_bl.append(l)
        selected_cols = list(df.drop(vis_cols, axis=1).columns) + vis_bl
        return df[selected_cols]
        
    @staticmethod
    def show_saved_csv(fpath="W:\\MSc-AIDS-UoH\\Trimester_3\\clean_pds\\"):
        """display path and names of all csv files saved at:
        W:\\MSc-AIDS-UoH\\Trimester_3\\clean_pds\\
        Return
        csv_files: dict containing {file_loc:fnames_list}"""
        
        csv_files = dict()
        data_dir = fpath
        # os.getcwd()
        csv_files[data_dir] = os.listdir(data_dir)
        return csv_files
        
    @staticmethod
    def reverse_dict_items(dd:dict):
        """reverse the order of dictionary items from key:value to value:key"""
        
        return {v:k for k,v in dd.items()}
    
    @staticmethod
    def outlier_detection_iqr(arr, n_iqr=1.5, precision=2):
        """Detect Outliers in arr (1-D array) based on n_iqr * IQR.
        Where IQR (Inter-quartile range) is the middle 50% of sorted 
        IQR of arr = Q3 (75th percentile) - Q1 (25th percentile)
        Outliers are data points below Q1 - (n_iqr * IQR )
        or above Q3 + (n_iqr * IQR)
        ie., where arr < Q1 - (n_iqr*IQR) or arr > Q3 + (n_iqr*IQR)"""
        
        def detect_outliers(num, lower, upper):
            if (num < lower) or (num > upper):
                return num
            
        x = np.sort(arr)
        q1, q3 = np.percentile(x, q=[25, 75], interpolation='midpoint')
        iqr = q3 - q1
        coef = n_iqr*iqr

        llim, ulim = np.round(q1 - coef, precision), np.round(q3 + coef, precision)
        outl_ser = pd.Series(x).apply(lambda n: detect_outliers(n, llim, ulim))
        outliers = outl_ser.loc[outl_ser.notnull()]
        return {'lower_limit':llim, 'upper_limit':ulim, 'n_outliers':len(outliers), 'outliers':outliers}
    
    @staticmethod
    def outlier_detection_kmeans(df, n_clusters=3, top_n=10):
        """detect outliers using KMeans"""
        
        X = copy.deepcopy(df)
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        
        # determine the distance from each data point in df to the closest cluster centroid:
        distance = [(min(distance), i) for i,distance in enumerate(kmeans.transform(X))]
        
        # determine the top_n most distant outliers:
        indices_of_outliers = [row[1] for row in sorted(distance, key=lambda row: row[0], reverse=True)][:top_n]
        return X.loc[indices_of_outliers]
        
    @staticmethod
    def train_PCA(df, perc_components=0.95):
        """Reduce the dimension a dataframe using a PCA model.
        principal components explain a proportion of the variance
        of the given input
        perc_components: fraction of the df variance you want explained 
        by the output
        
        NOTE: Rescale the input to between (0, 1) BEFORE 
        using PCA
        
        Return 
        (compressed_df, PCA_comps)"""
        
        if not isinstance(perc_components, float):
            raise ValueError('please input the percentage of the input variance you want explained by the components')
        
        if (perc_components > 1.0) or (perc_components < 0):
            raise ValueError('please input a percentage of within range (0, 1)')
        
        np.random.seed(1)
        pca = s_dec.PCA(perc_components, random_state=1)
        df_pca = pd.DataFrame(np.round(pca.fit_transform(df), 2))
        cc = {i: f'Component_{i+1}' for i in range(len(pca.components_))}
        df_pca.columns = cc.values()
        
        pca_comp = pd.DataFrame(np.round(pca.components_, 2), columns=pca.feature_names_in_)
        expl_var =  pd.DataFrame({'explained_variance_perc': np.round(pca.explained_variance_ratio_, 2)})
        pca_comp = pd.concat([pca_comp, expl_var], axis=1)
        
        print(f"These {len(pca.components_)} components explains {np.round(100*pca_comp['explained_variance_perc'].sum(), 2)}% of input variance\n")
        print(f'The variance of each component is given as:\n{100*np.cumsum(pca.explained_variance_ratio_).round(2)}')
        
        return df_pca, pca_comp
    
    @staticmethod
    def generate_cagrepeat_guide(df, get_caglow=True):
        """generate a dict guide for caglow and caghigh series in df
        Return
        cagrepeat_guide: dict guide of caglow if get_caglow is True
        else caghigh dict guide"""
        
        cls = MSc_Proj()
        def change_key_name(dd: dict, old_key_name: str, new_key_name: str) -> dict:
            """change a key in a dictionary.
            The secret to iterate over a copy of the dictionary
            while making necessary changes in the original one.
            :param dd: the dictionary whose key needs to be changed.
            :param old_key_name: the key that is going to be deleted.
            :param new_key_name: the key that is replacing the old key name
            :return
            a dictionary with old_key_name deleted, and new_key_name added.

            Example:
            students = {'name': ['osagie', 'edoghogho', 'ug', 'omosede'],
                    'sex': ['male', 'female', 'female', 'female'],
                    'city': ['hull', 'toronto', 'london', 'berlin']}

            new_key = 'gender'

            print(change_key_name(students, 'sex', new_key))

            Output:
            {'name': ['osagie', 'edoghogho', 'ug', 'omosede'], 
            'city': ['hull', 'toronto', 'london', 'berlin'], 
            'gender': ['male', 'female', 'female', 'female']}"""

            for k in dd.copy():  # iterate over a copy of the dictionary
                if k.lower() == old_key_name.lower():
                    dd[new_key_name] = dd.pop(k)
            return dd

        cols = ['caglow', 'caghigh']
        # replace >28 in caglow and >70 in caghigh with 29 and 71 respectively
        # and cast both series to int
        cagrepeat_num = cls.replace_value_with(df[cols], replacement_guide={'>28':29, '>70':71}).astype(int)
        unq_cag = cagrepeat_num.apply(pd.Series.unique).apply(np.sort)
    #     print(unq_cag)

        result = unq_cag.apply(lambda x: {str(i):i for i in x})
        result[cols[0]] = change_key_name(result[cols[0]], '29', '>28')
        result[cols[1]] = change_key_name(result[cols[1]], '71', '>70')
        if get_caglow:
            return result[cols[0]]
        return result[cols[1]]

    @staticmethod
    def percentage_per_class(df, freq_colname:str, catg_colname:str, precision=2):
        """compute percentage of freq_colname per catg_colname's category"""
        
        cols = [catg_colname, freq_colname]
        denom = df[cols].groupby(cols[0]).sum().rename(columns={cols[-1]:'denom'})
        calc = pd.merge(df[cols], denom, on=cols[0])
    #     display(calc)
        return pd.concat([df, np.round(calc.apply(lambda row: row[freq_colname]/row['denom'], axis=1), precision)],
                         axis=1).rename(columns={0:'perc'}).drop(freq_colname, axis=1)
    
    @staticmethod
    def combine_multiple_columns(df, *multi_cols:str, output_name='combo_variable'):
        """collapse multiple columns into one column"""
        
        def collapsed(row, mult_vars):
            result = ""
            for i in range(len(mult_vars)):
                if i == 0:
                    result += f"{row[mult_vars[i]]}"
                else:
                    result += f"_{row[mult_vars[i]]}"
            return result
        
        if not multi_cols:
            multi_cols = df.columns
        multi_cols = list(multi_cols)
        collapsed_var = pd.Series(df.apply(lambda row: collapsed(row, multi_cols), axis=1),
                                  name=output_name)
        return pd.concat([collapsed_var, df.drop(multi_cols, axis=1)], axis=1)
        
    @staticmethod
    def cleanup_participation(df, startdate_col:str, enddate_col:str, duration_output_name:str,
                          drop_colnames=None, rename_col_guide: dict=None):
    
        nan_placeholder = -999
        
        # derive duration from end and start date
        date_df = df[[startdate_col, enddate_col]].fillna(nan_placeholder)
        # display(date_df)
        end = date_df.apply(lambda row: 
                            row[startdate_col] if row[enddate_col] == nan_placeholder else row[enddate_col],
                           axis=1)
        duration = pd.Series(end - df[startdate_col], name=duration_output_name)
        
        # drop unimportant variables
        new_df = pd.concat([df.drop(drop_colnames, axis=1), duration], axis=1)
        
        # rename columns
        new_df = new_df.rename(columns=rename_col_guide)
        
        return new_df
    
    @staticmethod
    def convert_rows_to_columns(df, unique_identifier:str, shared_identifier:str, var_prefix:str=None, output_multi_lvl=False):
            """generate columns from row values per unique_identifier
            unique_identifier: primary identifier column
            shared_identifier: its unique values will each form a new header"""
            
            cls = MSc_Proj()
            unq = df[shared_identifier].unique()
            col_rename = {u: f"{var_prefix}_{u}" if var_prefix else u for u in unq}
            # print(col_rename)
            new_df = pd.DataFrame.pivot(df, index=unique_identifier, columns=shared_identifier).reset_index().rename(columns=col_rename)
            if output_multi_lvl:
                return new_df
            return cls.convert_multilevel_columns_to_single(new_df, unique_identifier)
            
    @staticmethod
    def convert_multilevel_columns_to_single(df, uniq_id):
        """convert a multi-level column dataframe to a single-level dataframe"""
        
        new_colnames = [f"{c1}_{c2}" if c1.lower() != uniq_id.lower() else c1 for c1, c2 in df.columns]
        new_df = df.droplevel(level=1, axis=1)
        new_df.columns = new_colnames
        return new_df
        
    @staticmethod
    def convert_column_to_row(df, primary_id:str, combo_column_name:str, value_name:str):
        """generate rows from columns"""
        
        return pd.DataFrame.melt(df, id_vars=primary_id, var_name=combo_column_name, value_name=value_name)
    
    @staticmethod
    def impute_null_values(df: 'pd.DataFrame', pivot_cols: list, target_col: str, stat_used: str='mean'):
        """Impute the null values in a target feature using aggregated values
        (eg mean, median, mode) based on pivot features
        :param df:
        :param pivot_cols:
        :param target_col:
        :param stat_used: {'mean', 'mode', 'median'}
        :return: impute_guide: 'a pandas dataframe'"""
        
        cls = MSc_Proj()
        
        if ('object' in str(df[target_col].dtypes)) or ('bool' in str(df[target_col].dtypes)):
            stat_used = 'mode'

        if str.lower(stat_used) == 'mean':
            impute_guide = cls.pivot_mean(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'mode':
            impute_guide = cls.pivot_mode(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'median':
            impute_guide = cls.pivot_median(df, pivot_cols, target_col)
        # fill null values with means
        null_ind = df.loc[df[target_col].isnull()].index

        piv_rec = pd.merge(left=df.loc[null_ind, pivot_cols],
                           right=impute_guide,
                           how='left', on=pivot_cols)
        # fill all lingering null values with general mean
        if piv_rec[target_col].isnull().sum():
            piv_rec.loc[piv_rec[target_col].isnull(), target_col] = impute_guide[target_col].mode().values[0]

        piv_rec.index = null_ind

        return pd.concat([df.loc[df[target_col].notnull(), target_col],
                          piv_rec[target_col]],
                         join='inner', axis=0).sort_index()
    
    @staticmethod
    def pivot_mode(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the mode (i.e the highest occurring) target_col
        value per combination of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")

        freq_df = df.loc[df[target_col].notnull(), pivot_cols + [target_col]].value_counts().reset_index()
        return freq_df.drop_duplicates(subset=pivot_cols).drop(labels=[0], axis=1)
        
    @staticmethod
    def pivot_mean(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the average target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")
        
        dec_places = 4
        targ_dtype = str(df[target_col].dtypes)
        
        if ('int' in targ_dtype):
            dec_places = 0
            targ_dtype = str(df[target_col].dtypes)[:-2]
            
        elif ('float' in targ_dtype):    
            sample = str(df.loc[df[target_col].notnull(), target_col].iloc[0])
            dec_places = len(sample.split('.')[-1])
            targ_dtype = str(df[target_col].dtypes)[:-2]
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).mean().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)
        
    @staticmethod
    def pivot_median(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the median target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")
        
        dec_places = 4
        targ_dtype = str(df[target_col].dtypes)
        
        if ('int' in targ_dtype):
            dec_places = 0
            targ_dtype = str(df[target_col].dtypes)[:-2]
            
        elif ('float' in targ_dtype):    
            sample = str(df.loc[df[target_col].notnull(), target_col].iloc[0])
            dec_places = len(sample.split('.')[-1])
            targ_dtype = str(df[target_col].dtypes)[:-2]
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).median().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)
    
    @staticmethod
    def get_ttest_pvalue_from_array(X1_set, X2_set, 
                                alternative_hypothesis='two-sided', precision=None):
            """compute tstatistics and pvalue of two arrays X1 and X2.
            alternative_hypothesis: {'two-sided', 'less', 'greater'}
            Return
            result_dict: {'t_statistic':t, 'pvalue':p}"""
            
            def check_equal_var(var1, var2):
                """determine whether to use student (equal variance) 
                or Welch (unequal variance) t-test.
                equal variance = (larger variance/smaller variance < 4) or (larger std/smaller std < 2)
                unequal variance = (larger var/smaller var >= 4)"""
                
                top, bot = max([var1, var2]), min([var1, var2])
                return ((top+1)/(bot+1)) < (4+1) # add 1 to cancel out ZeroDiv
            
            var1, var2 = np.var(X1_set, ddof=1), np.var(X2_set, ddof=1)

            if len(np.shape(X1_set)) > 1: # multiple features
                ev = [check_equal_var(ix1, ix2) for ix1, ix2 in zip(var1, var2)]
            else:
                ev = check_equal_var(var1, var2)
                if ev:
                    print("\nEqual Variance Detected! -> Student T-Test\n")
                else:
                    print("\nUnequal Variance Detected! -> Welch Test:\n")
            if not precision:
                t, p = ttest_ind(X1_set, X2_set, equal_var=ev, alternative=alternative_hypothesis)
            else:
                t, p = np.round(ttest_ind(X1_set, X2_set, equal_var=ev, 
                                          alternative=alternative_hypothesis), precision)
            return {'t_statistic':t, 'pvalue':p}

    @staticmethod
    def get_ttest_pvalue_from_stats(X1_mean, X1_std, X1_count, X2_mean, X2_std, X2_count,
                                    alternative_hypothesis='two-sided', precision=None):
            """compute tstatistics and pvalue of two arrays X1 and X2 from the mean,
            std, and sample size.
            alternative_hypothesis: {'two-sided', 'less', 'greater'}
            Return
            result_dict: {'t_statistic':t, 'pvalue':p}"""
            
            def check_equal_var(var1, var2):
                """determine whether to use student (equal variance) 
                or Welch (unequal variance) t-test.
                equal variance = (larger variance/smaller variance < 4) or (larger std/smaller std < 2)
                unequal variance = (larger var/smaller var >= 4)"""
                
                top, bot = max([var1, var2]), min([var1, var2])
                return (top/bot) < 4
            
            ev = check_equal_var(X1_std, X2_std)
            if ev:
                print("\nEqual Variance Detected! -> Student T-Test\n")
            else:
                print("\nUnequal Variance Detected! -> Welch Test:\n")
            if not precision:
                t, p = ttest_ind_from_stats(mean1=X1_mean, std1=X1_std, nobs1=X1_count,
                                            mean2=X2_mean, std2=X2_std, nobs2=X2_count,
                                            equal_var=ev, alternative=alternative_hypothesis)
            else:
                t, p = np.round(ttest_ind_from_stats(mean1=X1_mean, std1=X1_std, nobs1=X1_count,
                                            mean2=X2_mean, std2=X2_std, nobs2=X2_count,
                                            equal_var=ev, alternative=alternative_hypothesis), altprecision)
            return {'t_statistic':t, 'pvalue':p}
    
    @staticmethod
    def report_a_significance(X1_set, X2_set, n_deg_freedom=1, X1_name='X1', X2_name='X2', seed=None, balance=True):
        """Test for statistical significant difference between X1_set and X2_set
        at 99% and 95% Confidence.
        X1_set: 1D array of observations
        X2_set: 1D array of observations."""
        
        cls = MSc_Proj()
        
        def get_min_denom(n1, n2):
            return min([n1, n2])
        
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        # to ensure reproducibility
        if not seed:
            seed = 1
        np.random.seed(seed)
        
        samp_sizes = {X1_name: pd.Series(X1_set), 
                      X2_name: pd.Series(X2_set)}
        
        print(f'\n\nHYPOTHESIS TEST FOR:\n{X1_name} > {X2_name}\n')
        
        # use to compare single values
        if len(samp_sizes[X1_name]) == 1:
            total_X1, X1_mean, X1_std = samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0]**0.5
            total_X2, X2_mean, X2_std = samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0]**0.5
        else:
            X1_size, X2_size = len(X1_set), len(X2_set)
            print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')
            
            # check if sample sizes are unequal
            if detect_unequal_sizes(X1_size, X2_size):
                print("Unequal Sample Sizes Detected!!\n")
                if balance:
                    print("\n....DOWNSAMPLING RANDOMLY....\n")
                    min_size = get_min_denom(X1_size, X2_size)
                    max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
                    # downsampling: 
                    # randomly generate min_size indexes for max_samp_name
                    rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size, random_state=seed)
                    # select only random min_size indexes for max_samp_name set
                    samp_sizes[max_samp_name] = samp_sizes[max_samp_name].iloc[rand_indexes]
                    X1_size, X2_size = len(samp_sizes[X1_name]), len(samp_sizes[X2_name])
                    print(f'ADJUSTED SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                          f'{X2_name}: {X2_size}\n\n')
            total_X1, X1_mean, X1_std = cls.compute_mean_std(samp_sizes[X1_name], X1_size, n_deg_freedom)
            total_X2, X2_mean, X2_std = cls.compute_mean_std(samp_sizes[X2_name], X2_size, n_deg_freedom)
        
        null_hypo = np.round(X1_mean - X2_mean, 4)
        pooled_std = cls.compute_pstd(X1_std, X2_std)

        print(f'{X1_name}:\n Total = {total_X1}\n Average = {X1_mean}\n Standard deviation = {X1_std}\n\n' +
              f'{X2_name}:\n Total = {total_X2}\n Average = {X2_mean}\n Standard deviation = {X2_std}\n\n' +
              f'MEAN DIFFERENCE = {null_hypo}\n' +
              f'POOLED STD = {pooled_std}\n\n')
        
        print(f'HYPOTHESIS TEST:\nIs {X1_mean} significantly HIGHER THAN {X2_mean}?\n' +
             f'BASED ON the chosen level of significance\nIs the difference {null_hypo} > 0?\n')

        # check for both 99% and 95% confidence level
        # Meaning, is the difference between both figures greater than 
        # 3 pooled std and 2 pooled std respectively

        alpha = 0.01
        test_result = cls.compute_test(pooled_std, alpha)
        if null_hypo > test_result[1]:
            return print(f'At {test_result[0]}, REJECT the null hypothesis!\n {null_hypo} is greater than {test_result[1]}\n')
        else:
            test_result = cls.compute_test(pooled_std)
            if null_hypo > test_result[1]:
                return print(f'At {test_result[0]}, REJECT the null hypothesis!\n{null_hypo} is greater than {test_result[1]}\n')
        print(f'Do NOT reject the null hypothesis\n{null_hypo} is less than or equal to {test_result[1]}')
        
    @staticmethod
    def calc_deg_freedom(denom, n_deg):
        """compute degrees of freedom."""
        
        return denom - n_deg
        
    @staticmethod
    def compute_mean_std(arr, denom, n_deg):
        """compute sum, mean, stdev of array using 
        the given denominator and degrees of freedom"""
        
        cls = MSc_Proj()
        
        total = np.sum(arr)
        avg = np.round(total/denom, 4)
        deg_freedom = cls.calc_deg_freedom(denom, n_deg)
        sumsq = np.sum((arr - avg)**2)
        stdv = np.sqrt(sumsq/deg_freedom).round(4)
        return (total, avg, stdv)
        
    @staticmethod
    def index_generator(sample_size: 'array', n_index=1, random_state=1):
        """Randomly generate n indexes.
        :Return: random_indexes"""

        import random

        def select_from_array(sample_array, n_select=1, random_state=1):
            np.random.seed(random_state)
            return random.choices(population=sample_array, k=n_select)
        
        indexes = range(0, sample_size, 1)

        return select_from_array(indexes, n_index, random_state)
        
    @staticmethod
    def compute_pstd(stdv_1, stdv_2):
        """Compute pooled standard devs from two stdevs"""
        
        return round(np.sum([stdv_1**2, stdv_2**2])**0.5, 4)
    
    @staticmethod
    def compute_test(pooled_stdv, at_alpha=0.05):
        """Compute test sigma at specified significance with pooled_std.
        at_alpha: significance
        pooled_std: pooled standard deviation
        Return: (confidence, test_sigma)"""
        
        sig_to_conf = {0.05: (2, '95% confidence'),
                      0.01: (3, '99% confidence')}
        test_sigma = round(sig_to_conf[at_alpha][0] * pooled_stdv, 4)
        return (sig_to_conf[at_alpha][1], test_sigma)
    
    @staticmethod
    def equalize_categories(left, right, grouper_colname:str, freq_colname:str):
        """include zero frequencies for missing categories between two dfs:
        Return
        result: keys are (left, right) containing df with 0 in place of missing categories"""
        
        def merge_categories(catg1: 'pd.Series', catg2: 'pd.Series'):
            """create a common list of categories"""
            
            unq1, unq2 = list(catg1.unique()), list(catg2.unique())
            merg = pd.Series(sorted(set(unq1 + unq2)))
            return merg
        
        result = dict()
        common_categs = merge_categories(left[grouper_colname], right[grouper_colname])
        common_categs.name = grouper_colname
        common_categs = pd.DataFrame(common_categs)
        result['left'] = pd.merge(common_categs, left, on=grouper_colname, how='left')
        result['left'][freq_colname] = result['left'][freq_colname].fillna(int(0))
        result['right'] = pd.merge(common_categs, right, on=grouper_colname, how='left')
        result['right'][freq_colname] = result['right'][freq_colname].fillna(int(0))
        return result
        
    @staticmethod
    def get_gastroint_issues(mhbodsys_ser: 'pd.Series'):
        """generate participants with gastrointestinal illnesses
        gi illness is true where mhbodsys == 7
        Return
        has_gastroint_illness: 1 (yes) or 0 (no)""" 
        
        return mhbodsys_ser.apply(lambda x: 1 if x == 7 else 0)

    @staticmethod
    def bmdiff_band_guide(main_df, key_as_integer=False):
        """generate dict guide for bmi diff band column based on the values of
        columns ['bmi_diff_band', 'bmi_diff_class'] in main_df
        Return
        bmi_diff_guide: dict containing {'bmi_diff_band': 'bmi_diff_class'} if key_as_integer is False
        else the reverse is the case"""
        
        cols = ['bmi_diff_band', 'bmi_diff_class']
        rec = dict(main_df[cols].to_records(index=False))
        if not key_as_integer:
            return {k:v for v,k in sorted([(v,k) for k,v in rec.items()])}
        return dict(sorted([(v,k) for k,v in rec.items()]))

    @staticmethod
    def get_interval_freq(continuous_ser, n_groups=5, bin_width=None, precision=2, output_colnames=None):
            """generate interval frequencies for a continuous series.
            NOTE: categories/n_labels = num_intervals - 1
            Return
            bmi_change_interval: dataframe containing (intervals_as_int, intervals_as_str)"""
            
            cls = MSc_Proj()
            
            def classify_val(val:'number', interval_guide:dict):
                for ix, invls in interval_guide.items():
                    lower_bound, upper_bound = float(invls.split()[0][1:]), float(invls.split()[-1][:-1])
                    if (val >= lower_bound and val < upper_bound):
                        return ix 
            
            x = pd.Series(continuous_ser)
            min_val, max_val = np.percentile(x, [0, 100])
            labes_dict = cls.generate_interval_labels(min_val, max_val, n_groups, bin_width, precision)
            invl_ix = x.apply(lambda x: classify_val(x, labes_dict))
            invl_labe = invl_ix.map(labes_dict)
            if not output_colnames:
                output_colnames = ['bmi_diff_class', 'bmi_diff_band']
            return pd.concat([invl_ix, invl_labe], axis=1, keys=output_colnames)
            
    @staticmethod
    def generate_interval_labels(min_val, max_val, n_groups=5, bin_width=None, precision=2):
        """generate labels from lower and upper bounds of interval values.
        NOTE: categories/n_labels = num_intervals - 1
        Return
        label_dict: dictionary containing integer index: label"""
        
        def push_boundary(upper_lim, interval):
            """to make sure that the upper boundary is included"""
            
            remainder = upper_lim%interval
            if remainder != 0:
                return (interval - remainder) + upper_lim
            return upper_lim
        
        num_range = np.round(np.linspace(min_val, max_val+0.1, n_groups-1), precision)
        
        if bin_width:
            if max_val < 0:  # upper boundary is a negative value
                pmax = -1*push_boundary(np.abs(max_val), bin_width) + (bin_width/10)
                num_range = np.arange(start=min_val, stop=pmax, step=bin_width)
            elif min_val < 0:  # lower boundary is a negative value
                # include negative lower boundary
                nmax = push_boundary(np.abs(min_val), bin_width) + (bin_width/10)
                # negative range of values, excluding the first value (-0)
                nmax_range = -np.arange(start=0, stop=nmax, step=bin_width)[1:]
                # include positive upper boundary
                pmax = push_boundary(max_val, bin_width) + (bin_width/10)
                # positive range of values
                pmax_range = np.arange(start=0, stop=pmax, step=bin_width)
                # merge negative and positive range of values
                num_range = np.sort(np.append(nmax_range, pmax_range))
                
            else:
                num_range = np.round(np.arange(min_val, max_val+0.1, bin_width), precision)
            
        return {i:f'[{num_range[i-1]} to {num_range[i]})' for i in range(1, len(num_range))}
    
    @staticmethod
    def get_hdcat_labes(hdcat_series):
        """generate appropriate labels for hd categories integers"""
        
        cls = MSc_Proj()
        return hdcat_series.map(cls.hd_status_guide)
    
    @staticmethod
    def get_hd_categories(participation_df):
        """generate subjids of participants of various hd status.
        hdcat = {2:'pre-manifest', 3:'manifest', 4:'genotype -ve', 5:'family controls'}
        Return
        hd_participants: dict containing unique subject ids in various hd categs
        keys include: (pre_manifest, manifest, gtype_negative, fam_controls)"""
        
        cls = MSc_Proj()
        hd0 = cls.get_hdcat_labes(participation_df['hdcat_0'])
        hdl = cls.get_hdcat_labes(participation_df['hdcat_l'])
        
        def manifest_hd_participants(participation_df):
            """select participants with manifest HD at some point during their participation
            hdcat_0: participant hd status at enrolment
            hdcat_l: participant hd status at latest (most recent) visit
            hd_categs = {2:'pre-manifest', 3:'manifest', 4:'genotype -ve', 5:'family controls'}
            Return
            hd_participants: dataframe of participants with manifest hd at hdcat_0 or hdcat_l"""

            par = participation_df.loc[((hd0 == '3. manifest') |
                                        (hdl == '3. manifest'))]
            return tuple(par['subjid'].unique())
        
        def premanifest_hd_participants(participation_df):
            par = participation_df.loc[((hd0 == '2. pre-manifest') |
                                        (hdl == '2. pre-manifest'))]
            return tuple(par['subjid'].unique())
        
        def gtype_negative_participants(participation_df):
            par = participation_df.loc[((hd0 == '4. genotype -ve') |
                                        (hdl == '4. genotype -ve'))]
            return tuple(par['subjid'].unique())
        
        def family_controls_participants(participation_df):
            par = participation_df.loc[((hd0 == '5. family controls') |
                                        (hdl == '5. family controls'))]
            return tuple(par['subjid'].unique())
        
        hd_participants = dict()
        hd_participants['2. pre_manifest'] = premanifest_hd_participants(participation_df)
        hd_participants['3. manifest'] = manifest_hd_participants(participation_df)
        hd_participants['4. gtype_negative'] = gtype_negative_participants(participation_df)
        hd_participants['5. fam_controls'] = family_controls_participants(participation_df)
        
        return hd_participants
        
    @staticmethod
    def hd_categorize_df(participation_df, df):
        """categorize participants in df according to their hd categories.
        Return
        df_categs_dict: dictionary containing df info of various types of participants.
        Keys include: ('pre_manifest', 'manifest', 'gtype_negative', 'fam_controls')"""
        
        cls = MSc_Proj()
        hd_categs_dict = cls.get_hd_categories(participation_df)
        df_categs_dict = dict()
        df_categs_dict['pre_manifest'] = df.loc[df['subjid'].isin(hd_categs_dict['pre_manifest'])]
        df_categs_dict['manifest'] = df.loc[df['subjid'].isin(hd_categs_dict['manifest'])]
        df_categs_dict['gtype_negative'] = df.loc[df['subjid'].isin(hd_categs_dict['gtype_negative'])]
        df_categs_dict['fam_controls'] = df.loc[df['subjid'].isin(hd_categs_dict['fam_controls'])]
        
        return df_categs_dict
    
    @staticmethod
    def run_apriori(df, use_cols=None, use_colnames=True, min_support=0.5):
        """run the apriori algorithm to mine patterns among variables in a 
        dataframe.
        NOTE:
        all numeric variables are EXCLUDED before running apriori
        Return 
        result: apriori results in dataframe format."""
        
        sel_cols = df.select_dtypes(exclude='number')
        if use_cols:
            sel_cols = sel_cols[use_cols]
            
        result = frequent_patterns.apriori(pd.get_dummies(sel_cols),
                                       use_colnames=use_colnames, min_support=min_support)
        result['n_itmesets'] = result['itemsets'].apply(lambda x: len(x))
        return result.sort_values('support', ascending=False)
        
    @staticmethod
    def compute_bmi(w, h):
        """calculate bmi w (in kg) / h^2 (in meters)"""
        
        return round(w/(h/100)**2, 1)
        
    @staticmethod
    def remove_nan_hw(df):
        """remove nan weight and height"""
        
        clean_bmi_index = df.loc[((df['height'].isna()) |
                                  (df['weight'].isna()) |
                                 (df['height'] > 9000) |
                                 (df['weight'] > 9000))].index
        return df.loc[~df.index.isin(clean_bmi_index)]
        
    @staticmethod
    def count_occurrences(df, count_colname:str =None, output_name='total_count', attach_to_df=False):
        """count occurrences per unique count_colname or unique combined categs"""
        
        if not count_colname:
            freq = df.value_counts().reset_index()
        else:
            freq = df.groupby(count_colname).size().reset_index()
        freq = freq.rename(columns={0:output_name})
        if attach_to_df and count_colname:
            return pd.merge(freq, df, on=count_colname)
        return freq
        
    @staticmethod
    def compare_variables(var1, var2, var1_name=None, var2_name=None, where_equal=True):
        """compare two variables and return where equal/unequal (if True/False)
        Return
        result: dataframe where series are equal/unequal"""
        
        if not var1_name:
            var1_name = var1.name
        if not var2_name:
            var2_name = var2.name
            
        df = pd.concat([pd.Series(var1, name=var1_name),
                        pd.Series(var2, name=var2_name)], axis=1)
        cond = (var1 == var2).astype(int)
        if where_equal:
            return df.loc[cond == 1]
        return df.loc[cond == 0]
        
    @staticmethod
    def get_from_df(df, col1, col1_is, col2=None, col2_is=None, col3=None, col3_is=None):
        """return filtered view of dataframe"""
        
        def show_with_one_cond(df, col1, col1_is):
            """return filtered view of dataframe"""
            if df is None:
                if isinstance(col1, pd.Series):
                    if isinstance(col1_is, (tuple, list)):
                        cond = (col1.isin(col1_is))
                    else:
                        cond = (col1 == col1_is)
                return col1.loc[cond]
            if isinstance(col1_is, (tuple, list)):
                cond = (df[col1].isin(col1_is))
            else:
                cond = (df[col1] == col1_is)
            return df.loc[cond]

        def show_with_two_cond(df, col1, col1_is, col2, col2_is):
            """return filtered view of dataframe"""
            
            result = show_with_one_cond(df, col1, col1_is)
            if isinstance(col2_is, (tuple, list)):
                cond = (result[col2].isin(col2_is))
            else:
                cond = (result[col2] == col2_is)
            return result.loc[cond]

        def show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is):
            """return filtered view of dataframe"""
            
            result = show_with_two_cond(df, col1, col1_is, col2, col2_is)
            if isinstance(col3_is, (tuple, list)):
                cond = (result[col3].isin(col3_is))
            else:
                cond = (result[col3] == col3_is)
            return result.loc[cond]
        
        if col2 is not None and col2_is is not None:
            
            if col3 is not None and col3_is is not None:
                return show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is)
            
            return show_with_two_cond(df, col1, col1_is, col2, col2_is)
        
        return show_with_one_cond(df, col1, col1_is)
    
    @staticmethod
    def load_participant_based(pds_folder_name:str=None):
        """load and return PARTICIPANT-BASED DATA FILES
        profile
        pharmacotx
        nonpharmacotx
        nutsuppl
        comorbid
        profile is the main file and all other files are connected to it via subjid.
        Return
        participant_data_dict: dictionary containing each dataframes with corresponding names as keys"""
        
        cls = MSc_Proj()
        if not pds_folder_name:
            pds_folder_name = os.getcwd()
        file_path_dict = cls.file_search(pds_folder_name, search_file_type='csv')
        participant_based = {}
        participant_based['profile'] = pd.read_csv(file_path_dict['profile.csv'], sep='\t')
        participant_based['pharm'] = pd.read_csv(file_path_dict['pharmacotx.csv'], sep='\t')
        participant_based['nonpharm'] = pd.read_csv(file_path_dict['nonpharmacotx.csv'], sep='\t')
        participant_based['nutsup'] = pd.read_csv(file_path_dict['nutsuppl.csv'], sep='\t')
        participant_based['comorbid'] = pd.read_csv(file_path_dict['comorbid.csv'], sep='\t')
        return participant_based

    @staticmethod
    def load_study_based(pds_folder_name: str=None):
        """load and return STUDY-BASED DATA FILES
        participation
        events
        
        participation is the main file and it is connected to profile via subjid
        events and visit-based files are connected to it via [subjid, studyid]
        Return
        study_data_dict: dictionary containing each dataframes with corresponding names as keys"""
        
        cls = MSc_Proj()
        if not pds_folder_name:
            pds_folder_name = os.getcwd()
        file_path_dict = cls.file_search(pds_folder_name, search_file_type='csv')
        study_based = {}
        study_based['participation'] = pd.read_csv(file_path_dict['participation.csv'], sep='\t')
        study_based['event'] = pd.read_csv(file_path_dict['event.csv'], sep='\t')
        return study_based

    @staticmethod
    def load_visit_based(pds_folder_name: str=None):
        """load and return VISIT-BASED DATA FILES
        enroll
        registry
        ad hoc
        assessment
        These are all connected to participation via [subjid, studyid]
        Return
        visit_data_dict: dictionary containing each dataframes with corresponding names as keys"""
        
        cls = MSc_Proj()
        if not pds_folder_name:
            pds_folder_name = os.getcwd()
        file_path_dict = cls.file_search(pds_folder_name, search_file_type='csv')
        visit_based = {}
        visit_based['enroll'] = pd.read_csv(file_path_dict['enroll.csv'], sep='\t', low_memory=False)
        visit_based['registry'] = pd.read_csv(file_path_dict['registry.csv'], sep='\t')
        visit_based['assessment'] = pd.read_csv(file_path_dict['assessment.csv'], sep='\t')
        visit_based['ad_hoc'] = pd.read_csv(file_path_dict['adhoc.csv'], sep='\t')
        return visit_based
        
    @staticmethod
    def get_cagrepeat_band(cag_ser, output_name='caghigh_band'):
        """categorize CAG repeat extensions into 4 classes
        0. normal: < 27
        1. intermediate: 27 <= x < 36
        2. reduced penetrance: 36 <= x < 40
        3. full penetrance: x >= 40"""
        
        return pd.Series(cag_ser.apply(lambda x: '0. normal' if x < 27 else
                                       '1. intermediate' if x >= 27 and x < 36 else
                                       '2. reduced penetrance' if x >= 36 and x < 40 else '3. full penetrance'),
                         name=output_name)
        
    @staticmethod
    def get_overall_bmi(per_vis):
        """use all bmi_diff values of each participant to compute the total bmi_diff for that participant.
        total_bmi_diff = sum of participant's bmi_diff
        overall_bmi_change = -1 if total_bmi_diff < 0; 1 if total_bmi_diff > 0; else 0
        Return
        pps_bmi_change"""
        
        cls = MSc_Proj()
        # first replace all baseline placeholder diffs (-999) with 0
        total_bmi_diff = pd.Series(cls.get_per_participant(per_vis, 
                                                 'bmi_diff_vis').fillna(0).replace(-999,
                                                                                   0).apply(lambda row: sum(row), axis=1),
                                   name='total_bmdiff')
        overall_bmi_change = pd.Series(total_bmi_diff.apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0),
                                       name='total_bmchange')
        return pd.concat([total_bmi_diff, overall_bmi_change], axis=1)
     
    @staticmethod
    def get_total_bmi_diff(enrl):
        """get the difference between the last and first BMI per pps
        Return
        total_bmi_change: dataframe consisting of (first_bmi, last_bmi, total_bmi_diff)"""
        
        cols = ['subjid', 'bmi']
        all_bmi = enrl[cols].groupby(cols[0]).agg(tuple).reset_index()
        # display(all_bmi)
        first_bmi = pd.Series(all_bmi[cols[-1]].apply(lambda x: x[0]),
                             name='first_bmi')
        last_bmi = pd.Series(all_bmi[cols[-1]].apply(lambda x: x[-1]),
                             name='last_bmi')
        total_change = pd.Series(last_bmi - first_bmi, name='total_bmi_diff')
        total_avg_bmi = pd.Series(all_bmi[cols[-1]].apply(np.mean), name='total_avg_bmi').round(1)
        first_last_bmi = pd.concat([all_bmi.drop('bmi', axis=1), total_avg_bmi, 
                                    first_bmi, last_bmi, total_change], axis=1)

        return pd.merge(enrl[cols[0]], first_last_bmi, on=cols[0], how='inner')
     
    @staticmethod
    def get_bmi_change(df):
        """compute change between baseline bmi and last bmi measurements.
        bmi_diff: previous bmi - current bmi
        bmi_change: {-999:first visit, -1:decrease, 0:no change, 1:increase}
        Return
        bmi_change_df: subjid, bmi_diff, bmi_change"""
        
        def bmi_diff_per_subject(bmi_per_subject: list):
            """compute difference between baseline bmi and last bmi per subject id"""

            # bmi_diff = np.round(np.diff(np.array(bmi_per_subject)), 1)
            # result = [-999]
            # result.extend(list(bmi_diff))
            
            if len(bmi_per_subject) > 1:
                return bmi_per_subject[-1] - bmi_per_subject[0]
            else:
                return -999
        
        def bmi_changes(bmi_diff_per_subject:list):
            """determine direction of change in bmi
            -1 if < 0; 1 if > 0 else 0"""
            
            return bmi_diff_per_subject if bmi_diff_per_subject == -999 else -1 if bmi_diff_per_subject < 0 else 1 if bmi_diff_per_subject > 0 else 0

        cols = ['subjid', 'bmi']
        unq_bmi = df[cols].groupby('subjid').agg(list).reset_index()
        bmi_diff = unq_bmi['bmi'].apply(bmi_diff_per_subject)
        bmi_changes = bmi_diff.apply(bmi_changes)
#         display(bmi_diff, bmi_changes, unq_bmi[cols[0]])
        bmi_df = pd.concat([unq_bmi[cols[0]], pd.Series(bmi_diff, name='bmi_diff'), 
                          pd.Series(bmi_changes, name='bmi_change')], axis=1)
        # expand rows from aggregated list
        # bmi_combo = bmi_df.explode(['bmi_diff', 'bmi_change']).reset_index(drop=True).drop(['subjid', 'bmi'], axis=1)
        # bmi_combo['bmi_diff'] = np.round(bmi_combo['bmi_diff'].astype(float), 1)
        return pd.merge(df[cols[0]], bmi_df, on=cols[0]).drop(['subjid'], axis=1)
     
    @staticmethod
    def get_bmi_band(df, bmi_colname:str='bmi'):
        """return bmi in buckets.
        below normal (underweight) = bmi < 18.5
        normal bmi (healthy) = bmi between 18.5 and 24.9, 
        above normal (overweight) = bmi between 25 and 29.9
        obese = bmi between 30 and 39.9
        morbidly obese (severely obese) = bmi >= 40
        bmi_outcome: {-1:'underweight', 0:'normal', 1:'overweight'}
        Return
        bmi_discretized: df containing columns: (bmi_level, bmi_level_num, bmi_outcome)#, bmi_band_group)"""
        
        cls = MSc_Proj()
        lower_lim, upper_lim = 18.5, 24.9
        bmi_discretized = dict()
        # bmi_discretized['bmi_band'] = df['bmi'].apply(lambda x: '0-10' if float(x) <= 10 else '11-20' if float(x) > 10 and float(x) <= 20
                                                      # else '21-30' if float(x) > 20 and float(x) <= 30 
                                                      # else '31-40' if float(x) > 30 and float(x) <= 40
                                                      # else '41-50' if float(x) > 40 and float(x) <= 50
                                                      # else '51-60' if float(x) > 50 and float(x) <= 60
                                                      # else '61-70' if float(x) > 60 and float(x) <= 70
                                                      # else '>70')
        # guide = {'0-10':1, '11-20':2, '21-30':3, '31-40':4, '41-50':5, '51-60':6, '61-70':7, '>70':8}
        # bmi_discretized['bmi_band_group'] = bmi_discretized['bmi_band'].map(guide)
        
        if isinstance(df, pd.DataFrame):
            bmi_ser = df[bmi_colname]
        if isinstance(df, pd.Series):
            bmi_ser = df
            
        bmi_discretized['bmi_level'] = bmi_ser.apply(lambda x: '0. underweight' if float(x) < 18.5
                                                              else '1. normal' if float(x) >= 18.5 and float(x) < 25
                                                             else '2. overweight' if float(x) >= 25 and float(x) < 30
                                                             else '3. obese' if float(x) >= 30 and float(x) < 40
                                                             else '4. severely obese')
        
        # bmi_discretized['bmi_level_num'] = bmi_discretized['bmi_level'].map(cls.bmi_lvl_guide)
        bmi_discretized['bmi_outcome'] = bmi_discretized['bmi_level'].apply(lambda x: -1 if x == '0. underweight' else 0 if x == '1. normal' else 1)
        return bmi_discretized#[['bmi_level', 'bmi_outcome']]
     
    @staticmethod
    def get_age_band(df, age_colname:str =None, in_5yr_inv=False):
        """generate age buckets in decades or five-yr intervals
        In 5yr intervals:
        {1:'below 18', 2:'18-24', 3:'25-29', 4:'30-34', 5:'35-39', 6:'40-44', 7:'45-49', 8:'50-54', 
        9:'55-59', 10:'60-64', 11:'65-69', 12:'above 70'}
        
        In 10yr intervals:
        ['0. <30', '1. 30 - 39', '2. 40 - 49', '3. 50 - 59', '4. 60 - 69', '5. >=70']
        
        Return
        age_band: dataframe having discretized and bucketized ages"""
        
        cls = MSc_Proj()
        if not age_colname:
            age_colname = 'age'
        df[age_colname] = df[age_colname].apply(lambda x: 0 if x == '<18' else 71 if x == '>70' else x).astype(int)
        
        if in_5yr_inv:
            five_yr_int = df[age_colname].apply(lambda x: 1 if x == 0 else 2 if int(x) >= 18 and int(x) < 25 
                                               else 3 if int(x) >= 25 and int(x) < 30 else 4 if int(x) >= 30 and int(x) < 35
                                               else 5 if int(x) >= 35 and int(x) < 40 else 6 if int(x) >= 40 and int(x) < 45
                                               else 7 if int(x) >= 45 and int(x) < 50 else 8 if int(x) >= 50 and int(x) < 55
                                               else 9 if int(x) >= 55 and int(x) < 60 else 10 if int(x) >= 60 and int(x) < 65
                                               else 11 if int(x) >= 65 and int(x) < 70 else 12)
            five_yr_buckets = five_yr_int.map(cls.age_band['five_yr_band'])
            age_band = pd.concat([pd.Series(five_yr_int, name='age_group'),
                                  pd.Series(five_yr_buckets, name='age_bucket')], axis=1)
        else:
            # ten_yr_int = df[age_colname].apply(lambda x: 1 if x == '<18' else 2 if int(x) >= 18  and int(x) < 31
                                                # else 3 if int(x) >= 31 and int(x) < 41 else 4 if int(x) >= 41 and int(x) < 51
                                                # else 5 if int(x) >= 51 and int(x) < 61 else 6 if int(x) >= 61 and int(x) < 71
                                                # else 7)
            # ten_yr_buckets = ten_yr_int.map(cls.age_band['ten_yr_band'])
            
            ten_yr_buckets = df[age_colname].apply(lambda x: '0. <30' if x >= 0 and int(x) < 30 else '1. 30 - 39' if int(x) >= 30  and int(x) < 40
                                                else '2. 40 - 49' if int(x) >= 40 and int(x) < 50 else '3. 50 - 59' if int(x) >= 50 and int(x) < 60
                                                else '4. 60 - 69' if int(x) >= 60 and int(x) < 70 else '5. >69')
            # ten_yr_buckets = ten_yr_int.map(cls.age_band['ten_yr_band'])
            age_band = pd.Series(ten_yr_buckets, name='age_bucket')
        return age_band
     
    @staticmethod
    def replace_value_with(df: 'pandas df', replacement_guide:dict=None, colnames:list=None):
        """replace all occurrences of old value (key) in colnames with new value (value).
            if colnames is None, replace all occurrences of old value across entire
            df with new value.
            Return 
            new_df: dataframe with new values in place of old values"""

        def replace_value(df, old_value, new_value, colnames=None):
            """replace all occurrences of old value in colnames with new value.
            if colnames is None, replace all occurrences of old value across entire
            df with new value.
            Return 
            new_df: dataframe with new values in place of old values"""

            if not colnames:
                cols = tuple(df.columns)
            else:
                cols = tuple(colnames)

            new_df = pd.DataFrame(df)

            for c in cols:
                new_df[c] = new_df[c].apply(lambda x: new_value if x == old_value else  x)
            return new_df
        
        for i, (old_val, new_val) in enumerate(replacement_guide.items()):
            if i == 0:
                new_df = replace_value(df, old_val, new_val, colnames)
                continue
            new_df = replace_value(new_df, old_val, new_val, colnames)
        return new_df
    
    @staticmethod
    def corr_with_pearson(X, y, include_direction=False, scale_up=100, precision=2, top_n=None):
        """compute Pearson correlation for outcome y with X variables"""
        
        X, y = pd.DataFrame(X), pd.Series(y)
        if include_direction:
            return np.round(scale_up * X.corrwith(y), precision).sort_values(ascending=False).iloc[:top_n]
        return np.round(scale_up * X.corrwith(y).abs(), precision).sort_values(ascending=False).iloc[:top_n]
        
    @staticmethod
    def corr_with_kbest(X, y):
        """using sklearn.preprocessing.SelectKBest, quantify correlations
        between features in X and y.
        Return: correlation series"""
        
        X = pd.DataFrame(X)
        selector = s_fs.SelectKBest(k='all').fit(X, y)
        return pd.Series(selector.scores_, index=selector.feature_names_in_).sort_values(ascending=False)
    
    @staticmethod
    def select_corr_with_threshold(df: pd.DataFrame, thresh: float=0.5, report_only_colnames=False):
        """select only variables with correlation equal to or above the threshold value.
        Return:
        df_corr"""
        
        def count_thresh_corrs(row, thresh):
            """to count how many values in each row >= thresh"""
            
            count = 0
            for val in row:
                if abs(val) >= abs(thresh):
                    count += 1
            return count
        
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        print(f'Features with correlation of {round(100*thresh, 2)}% and above:\n')
        # number of values >= threshold value per row
        all_corrs = df.corr().round(4)
        selected_corr = all_corrs.apply(lambda val: count_thresh_corrs(val, thresh))
        correlated_vars = selected_corr[selected_corr > 1]
        if report_only_colnames:
            return list(correlated_vars.index)
        return all_corrs.loc[all_corrs.index.isin(list(correlated_vars.index)), list(correlated_vars.index)]
     
    @staticmethod
    def generate_aggregated_lookup(df, using_cols: list=None):
        """generate a lookup table using a subset of variables
        comprising a total accident count per category.
        Return:
        freq_lookup_df"""
        
        if not using_cols:
            using_cols = ('month', 'week_num', 'day_of_week', 'day_name', 'hour', 'day_num')
                                                     
        aggregate_df = df[using_cols].value_counts().sort_index().reset_index()
        aggregate_df.columns = aggregate_df.columns.astype(str).str.replace('0', 'total_participants')
        return aggregate_df
     
    @staticmethod
    def get_from_df(df, col1, col1_is, col2=None, col2_is=None, col3=None, col3_is=None):
        """return filtered view of dataframe"""
        
        def show_with_one_cond(df, col1, col1_is):
            """return filtered view of dataframe"""
            if df is None:
                if isinstance(col1, pd.Series):
                    if isinstance(col1_is, (tuple, list)):
                        cond = (col1.isin(col1_is))
                    else:
                        cond = (col1 == col1_is)
                return col1.loc[cond]
            if isinstance(col1_is, (tuple, list)):
                cond = (df[col1].isin(col1_is))
            else:
                cond = (df[col1] == col1_is)
            return df.loc[cond]

        def show_with_two_cond(df, col1, col1_is, col2, col2_is):
            """return filtered view of dataframe"""
            
            result = show_with_one_cond(df, col1, col1_is)
            if isinstance(col2_is, (tuple, list)):
                cond = (result[col2].isin(col2_is))
            else:
                cond = (result[col2] == col2_is)
            return result.loc[cond]

        def show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is):
            """return filtered view of dataframe"""
            
            result = show_with_two_cond(df, col1, col1_is, col2, col2_is)
            if isinstance(col3_is, (tuple, list)):
                cond = (result[col3].isin(col3_is))
            else:
                cond = (result[col3] == col3_is)
            return result.loc[cond]
        
        if col2 is not None and col2_is is not None:
            
            if col3 is not None and col3_is is not None:
                return show_with_three_cond(df, col1, col1_is, col2, col2_is, col3, col3_is)
            
            return show_with_two_cond(df, col1, col1_is, col2, col2_is)
        
        return show_with_one_cond(df, col1, col1_is)
     
    @staticmethod
    def rank_top_occurrences(df, ranking_col=None, top_n=3, min_count_allowed=1):
        """rank top n occurrences per given ranking column"""
        
        if not ranking_col:
            ranking_col = list(df.columns)[0]
        counts = df.value_counts().sort_values(ascending=False).reset_index()
        counts = counts.groupby(ranking_col).head(top_n)
        counts = counts.rename(columns={0:'total_participants'})
        if min_count_allowed:
            counts = counts.loc[counts['total_participants'] >= min_count_allowed]
        return counts.sort_values('total_participants', ascending=False).reset_index(drop=True)
        
    @staticmethod
    def create_label_from_ranking(df, exclude_last_col=True):
    
        if 'total_participants' in df.columns:
            df = df.drop('total_participants', axis=1)
        ranking_cols = list(df.columns)
        if exclude_last_col:
            ranking_cols = list(df.iloc[:, :-1].columns)
        labe, n_cols = [], len(ranking_cols)
        for i in range(len(df)):
            row_labe = ''
            for j in range(len(ranking_cols)):
                if j == n_cols - 1:
                    row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]}"
                    continue
                row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]} &\n"
                
            labe.append(row_labe)
        return pd.Series(labe, name='combined_variables', index=df.index)
    
    @staticmethod
    def give_percentage(arr, perc_total=None, precision=2):
        """output the percentage of each element in array
        arr: dataframe or series
        perc_total: total for calculating percentage of each value
        if perc_total is None, len(arr) is used instead
        Return
        perc_arr"""
        
        if not perc_total:
            return np.round(100*arr/len(arr), precision)
        return np.round(100*arr/perc_total, precision)
        
    @staticmethod
    def percentage_per_row(df, grouper_col:str=None, precision=2):
        """compute percentage per row of each unique grouper_col value
        Return
        result: df in percentage"""
        
        if grouper_col:
            new_df = df.set_index(grouper_col)
        else:
            new_df = df
        denom = new_df.sum(axis=1)
        return np.round(100 * new_df.div(denom, axis=0), precision)
        
    @staticmethod
    def get_subset_percentage(df, freq_col:str, sum_by_col:str, precision=2):
        """compute percentage per group of each unique sum_by_col value
        Return
        result: df including percentage"""
        
        # create aggregate sum for percentage
        cols = [sum_by_col, freq_col]
        summed = df[cols].groupby(cols[0]).sum().rename(columns={freq_col:'summed'}).reset_index()
        
        #attach aggregate value to each row
        new_df = pd.merge(df, summed, on=cols[0])
        new_df.loc[:, f'%{freq_col}'] = 100 * new_df.apply(lambda row: row[freq_col] / row['summed'], axis=1).round(precision)
        return new_df.drop('summed', axis=1)
        
    @staticmethod
    def plot_errorbars(x, y, yerror, axis, plot_title='Line Plot', title_size=15, capsize=2,
                  err_linewidth=0.5, errbar_color='black', shift=False):
        """plot errorbars on an axis.
        yerror can be int or array
        especially where there are multiple collection of means.
        use shift to push errorbar horizontally
        Return
        axis"""
        
        # move errorbars sideways
        trans1 = Affine2D().translate(-0.1, 0.0) + axis.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + axis.transData
        
        # plot errorbars on the background graph
        if not shift:
            axis.errorbar(x, y, yerr=yerror, fmt='none', capsize=capsize, elinewidth=err_linewidth,
                         ecolor=errbar_color, transform=trans1)
        else: # shift the errorbar horizontally
            axis.errorbar(x, y, yerr=yerror, fmt='none', capsize=capsize, elinewidth=err_linewidth,
                          ecolor=errbar_color, transform=trans2)
        return axis
        
    @staticmethod
    def plot_hist(x=None, y=None, condition_on=None, plot_title="A Histogram", bins=None, interval_per_bin: int=None, 
              bin_range=None, color=None, layer_type='default', x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150,
              stat='count', include_kde=False, savefig=False, fig_filename='histogram.png'):
        """A histogram plot on an axis.
        layer_type: {"layer", "dodge", "stack", "fill"}
        stat: {'count', 'frequency', 'probability', 'percent', 'density'}
        Return 
        axis"""
        
        cls = MSc_Proj()
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not bins:
            bins = 'auto'
        if str.lower(layer_type) == 'default':
            layer_type = 'layer'
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range, 
                                binwidth=interval_per_bin, color=color, multiple=layer_type,
                               stat=stat, kde=include_kde)
        else:
            sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range,
                         binwidth=interval_per_bin, color=color, multiple=layer_type, 
                         stat=stat, kde=include_kde, ax=axis)
        axis.set_title(plot_title, weight='bold')
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold')
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold')
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
            
        return axis
        
    @staticmethod
    def plot_strip(x, y, condition_on=None, x_labe_order=None, condition_order=None, plot_title='Strip Plot', title_size=14, 
               marker_size=5, color=None, paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, 
               figsize=(8, 4), dpi=200, orientation='v', rotate_xticklabe=None, rotate_yticklabe=None, alpha=None, 
               xy_ticklabe_size=6, savefig=False, fig_filename='stripplot.png'):
            """plot scatter graph for categorical vs numeric variables.
            Return: axis """
            
            cls = MSc_Proj()
            
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.stripplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, 
                                     size=marker_size, alpha=alpha, palette=paletter, color=color, orient=orientation)
            else:
                sns.stripplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, 
                              size=marker_size, alpha=alpha, ax=axis, palette=paletter, orient=orientation, color=color)
                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if orientation.lower() in 'vertical':
                axis.set_xticklabels(axis.get_xticklabels(), size=xy_ticklabe_size, rotation=rotate_xticklabe)
                axis.set_yticklabels(axis.get_yticks(), size=xy_ticklabe_size, rotation=rotate_yticklabe)
            elif orientation.lower() in 'horizontal':
                axis.set_xticklabels(axis.get_xticks(), size=xy_ticklabe_size, rotation=rotate_xticklabe)
                axis.set_yticklabels(axis.get_yticklabels(), size=xy_ticklabe_size, rotation=rotate_yticklabe)
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
                
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
        
    @staticmethod
    def plot_scatter(x, y, condition_on=None, plot_title='Scatter Plot', title_size=14, marker=None, color=None, 
                         paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, figsize=(8, 4), dpi=200,
                         rotate_xticklabe=False, alpha=None, savefig=False, fig_filename='scatterplot.png'):
            """plot scatter graph on an axis.
            Return: axis """
            
            cls = MSc_Proj()
            
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, 
                                      alpha=alpha, palette=paletter, color=color)
            else:
                sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                                alpha=alpha, ax=axis, palette=paletter, color=color)
                                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
                
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
        
    @staticmethod
    def plot_line(x, y, condition_on=None, plot_title='Line Plot', title_size=15, line_size=None, conf_intvl=None,
              paletter='viridis', legend_labe=None, show_legend_at=None, legend_size=7, marker=None, color=None, 
              xy_labe_size=8, x_labe=None, y_labe=None, rotate_xticklabe=0, axis=None, xlim=None, ylim=None, 
              xy_ticksize=7, figsize=(8, 4), dpi=200, savefig=False, fig_filename='linegraph.png'):
        """plot line graph on an axis.
        Return: axis """
        
        cls = MSc_Proj()
        
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, #legend=show_legend,
                               label=legend_labe,  palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, #legend=show_legend,
                        ax=axis, palette=paletter, color=color, label=legend_labe, ci=conf_intvl)
            
#         if not show_legend_at:
#             axis.legend().remove()
#         else:
#             axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        plt.xticks(ticks=axis.get_xticks(), fontsize=xy_ticksize, rotation=rotate_xticklabe, )
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_intervals(x, n_groups=5, include_right=True, precision=2, plot_title='A Histogram', x_labe=None,
                      y_labe='count', annotate=True, color=None, paletter='viridis', conf_intvl=None, include_perc=False, 
                      xy_labe_size=8, annot_size=6, use_bar=False, rotate_xticklabe=False, rotate_yticklabe=False, 
                       title_size=15, xlim: tuple=None, ylim: tuple=None, axis=None, xy_ticksize=7, perc_labe_gap=None, 
                       perc_total=None, h_labe_shift=0.1, v_labe_shift=0, perc_labe_color='black', bot_labe_color='blue',
                       reduce_barw_by=1, figsize=(8, 4), dpi=200, savefig=False, fig_filename='intervalplot.png'):
            """plot histogram on an axis.
            x: variable containing continuous values
            If include_perc is True, then perc_freq must be provided.
            :Return: axis """
            
            cls = MSc_Proj()
            bins = cls.get_interval_freq(x, n_groups, precision)
            freq = cls.count_occurrences(bins).sort_values(bins.columns[0])
            x, y = freq['bmi_diff_band'], freq['total_count']
            display(freq)
            
            if color:
                paletter = None
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                if not use_bar:
                    axis = sns.barplot(x=x, y=y, palette=paletter, color=color, ci=conf_intvl)
                else:
                    axis = sns.barplot(x=y, y=x, palette=paletter, color=color, ci=conf_intvl, orient='h')
            else:
                if not use_bar:
                    sns.barplot(x=x, y=y, ci=conf_intvl, 
                                palette=paletter, color=color, ax=axis)
                else:
                    sns.barplot(x=y, y=x, ci=conf_intvl, orient='h',
                            palette=paletter, color=color, ax=axis)
            
            axis.set_title(plot_title, weight='bold', size=title_size)
            
            if rotate_xticklabe:
                    axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
            if rotate_yticklabe:
                    axis.set_yticklabels(axis.get_yticklabels(), rotation=90)
            if x_labe:
                axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
            
            if xlim:
                axis.set_xlim(left=xlim[0], right=xlim[1])
            
            if ylim:
                axis.set_ylim(bottom=ylim[0], top=ylim[1])
                
            axis.set_title(plot_title, weight='bold', size=title_size)
            if not use_bar:
                axis.set_yticklabels(axis.get_yticks(), fontdict={'fontsize':xy_ticksize})
                axis.set_xticklabels(axis.get_xticklabels(), fontdict={'fontsize':xy_ticksize})
            else:
                axis.set_xticklabels(axis.get_xticks(), fontdict={'fontsize':xy_ticksize})
                axis.set_yticklabels(axis.get_yticklabels(), fontdict={'fontsize':xy_ticksize})
            
            if annotate: 
                cont = axis.containers
                for i in range(len(cont)):
    #                 print(len(cont))
                    axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                      weight='bold')
            for p in axis.patches:
                if not use_bar:
                    x = p.get_x()
                    w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
                    p.set_width(w)
                    p.set_x(center-w/2)
                else:
                    y = p.get_y()
                    bar_width, center = p.get_height()/reduce_barw_by, y+p.get_height()/2
                    p.set_height(bar_width)
                    p.set_y(center-bar_width/2)
                    
            if include_perc:
                if not perc_total:
                    perc_total = y.sum()
                
                for p in axis.patches:
                    x, y = p.get_xy()
                    if not use_bar:
                        bot_labe_pos = p.get_height()
                        perc = round(100 * p.get_height()/perc_total, precision)
                        labe = f'{perc}%'
                        perc_labe_pos = bot_labe_pos+perc_labe_gap
                        axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                                  size=annot_size, weight='bold')
                    else:
                        y_range = y.max() - y.min()
                        if not perc_labe_gap:
                            perc_labe_gap=y_range/1000 
                        perc = round(100 * p.get_width()/perc_total, precision)
                        labe = f'{perc}%'
                        perc_labe_pos = p.get_width()+perc_labe_gap
                        axis.text(perc_labe_pos, y-v_labe_shift, labe, color=perc_labe_color, 
                                  size=annot_size, weight='bold')
                        
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
    
    @staticmethod
    def adjust_axis(axis, plot_title='Plot Title', title_size=12, rotate_xticklabe=0, rotate_yticklabe=0, x_labe=None, precision=2,
                    y_labe=None, xy_labe_size=8, xlim=None, ylim=None, xy_ticksize=5, annotate=False, annot_size=6,  
                    reduce_barw_by=1, bot_labe_color='blue', include_perc=False, perc_total=None, perc_labe_color='black', 
                    show_legend_at:tuple=None, legend_size=7, perc_labe_gap=0, h_labe_shift=0, savefig=False, fig_filename='figplot.png'):
        """edit the setting of statistical plot diagram"""
            
        cls = MSc_Proj()
        
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)

        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])

        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})

        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(np.round(axis.get_yticks(), precision), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})

        # labels on columns
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
    #                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
        
        y_freq = list()
        # position of columns
        for p in axis.patches:
            x = p.get_x()
            w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
            p.set_width(w)
            p.set_x(center-w/2)
            y_freq.append(p.get_height())

        # percentage labels
        if include_perc:
            if not perc_total:
                perc_total = np.sum(y_freq)

            for p in axis.patches:
                x, y = p.get_xy()
                bot_labe_pos = p.get_height()
                perc = round(100 * p.get_height()/perc_total, precision)
                labe = f'{perc}%'
                perc_labe_pos = bot_labe_pos+perc_labe_gap
                axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                          size=annot_size, weight='bold')

        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_column(x, y, condition_on=None, condition_order=None, x_labe_order=None, plot_title='A Column Chart', title_size=15, x_labe=None, y_labe=None,
                    annotate=True, color=None, paletter='viridis', conf_intvl=None, include_perc=False, xy_labe_size=8, precision=2,
                    annot_size=6, perc_labe_gap=None, perc_total=None, h_labe_shift=0.1, perc_labe_color='black', bot_labe_color='blue', 
                    show_legend_at: tuple=None, legend_size=7, index_order: bool=True, rotate_xticklabe=0, xlim: tuple=None, ylim: tuple=None, 
                    axis=None, xy_ticksize=7, reduce_barw_by=1, figsize=(8, 4), dpi=200, savefig=False, fig_filename='columnplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = MSc_Proj()
        freq_col = pd.Series(y)
        
        if color:
            paletter = None
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order,
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, ci=conf_intvl, 
                        palette=paletter, color=color, ax=axis)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        
        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        
        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(np.round(axis.get_yticks(), precision), fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})
        
        y_range = y.max() - y.min()
        if not perc_labe_gap:
            perc_labe_gap=y_range/1000
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
        for p in axis.patches:
            x = p.get_x()
            w, center = p.get_width()/reduce_barw_by, x+p.get_width()/2
            p.set_width(w)
            p.set_x(center-w/2)
                    
        if include_perc:
            if not perc_total:
                perc_total = freq_col.sum()
                
            for p in axis.patches:
                x, y = p.get_xy()
                bot_labe_pos = p.get_height()
                perc = round(100 * p.get_height()/perc_total, precision)
                labe = f'{perc}%'
                perc_labe_pos = bot_labe_pos+perc_labe_gap
                axis.text(x-h_labe_shift, perc_labe_pos, labe, color=perc_labe_color, 
                          size=annot_size, weight='bold')
                    
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def plot_bar(x, y, condition_on=None, x_labe_order=None, condition_order=None, plot_title='A Bar Chart', title_size=15,
                 x_labe=None, y_labe=None, xy_labe_size=8, color=None, paletter='viridis', conf_intvl=None, include_perc=False, 
                 perc_total=None, annot_size=6, perc_labe_gap=10, v_labe_shift=0, perc_labe_color='black', bot_labe_color='blue',
                 precision=2, index_order: bool=True, rotate_yticklabe=0, annotate=False, axis=None, figsize=(8, 4), dpi=200,
                 xlim=None, xy_ticksize=7, show_legend_at:tuple=None, legend_size=7, reduce_barw_by=1, 
                 savefig=False, fig_filename='barplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = MSc_Proj()
        
        freq_col = x

        if color:
            paletter = None

        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, orient='h',
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, order=x_labe_order, hue_order=condition_order, ci=conf_intvl, 
                        palette=paletter, color=color, orient='h', ax=axis)

        if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
            
        if not show_legend_at:
            axis.legend().remove()
        else:
            axis.legend(loc=show_legend_at, prop={'size':legend_size})
            
        axis.set_title(plot_title, weight='bold', size=title_size)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=rotate_yticklabe, fontdict={'fontsize':xy_ticksize})
        axis.set_xticklabels(np.round(axis.get_xticks(), precision), fontdict={'fontsize':xy_ticksize})
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(container=axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
            if include_perc and perc_total is not None:
                for p in axis.patches:
                    x, y = p.get_xy()
                    perc = round(100 * p.get_width()/perc_total, precision)
                    labe = f'{perc}%'
                    perc_labe_pos = p.get_width()+perc_labe_gap
                    axis.text(perc_labe_pos, y-v_labe_shift, labe, color=perc_labe_color, 
                              size=annot_size, weight='bold')
        
        for p in axis.patches:
            y = p.get_y()
            bar_width, center = p.get_height()/reduce_barw_by, y+p.get_height()/2
            p.set_height(bar_width)
            p.set_y(center-bar_width/2)
                              
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_box(x=None, y=None, condition_on=None, condition_order=None, plot_title="A Boxplot", title_size=12, 
             orientation='vertical', x_labe_order=None, x_labe=None, y_labe=None, axis=None, paletter='viridis', 
             whiskers=1.5, color=None, figsize=(8, 4), dpi=150, show_legend_at=None, legend_size=8, y_lim=None,
             xy_labe_size=6, rotate_xticklabe=0, xy_ticksize=7, box_width=0.8, 
             savefig=False, fig_filename='boxplot.png'):
            """Draw a box distribution plot on an axis.
            Return: 
            axis"""
            
            cls = MSc_Proj()

            if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
                raise TypeError("x must be a pandas series or numpy array")
            elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
                raise TypeError("y must be a pandas series or numpy array")

            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.boxplot(x=x, y=y, hue=condition_on, hue_order=condition_order, order=x_labe_order,
                                   whis=whiskers, orient=orientation, width=box_width, color=color, palette=paletter)
            else:
                sns.boxplot(x=x, y=y, hue=condition_on, hue_order=condition_order, order=x_labe_order, 
                            whis=whiskers, orient=orientation, color=color, width=box_width, palette=paletter, ax=axis)

            if y_lim:
                axis.set_ylim(ymin=y_lim[0], ymax=y_lim[-1])
            axis.set_title(plot_title, size=title_size, weight='bold')
            axis.set_yticklabels(axis.get_yticks(), fontdict={'fontsize':xy_ticksize})
            axis.set_xticklabels(axis.get_xticklabels(), rotation=rotate_xticklabe, fontdict={'fontsize':xy_ticksize})

#             if x_labe:
            axis.set_xlabel(x_labe, weight='bold', size=xy_labe_size)
#             if y_labe:
            axis.set_ylabel(y_labe, weight='bold', size=xy_labe_size)

            if not show_legend_at:
                axis.legend().remove()
            else:
                axis.legend(loc=show_legend_at, prop={'size':legend_size})

            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
       
                            
    @staticmethod
    def plot_correl_heatmap(df, plot_title="A Heatmap", title_size=10, annot_size=6, xy_ticklabe_size=6,
                        xlabe=None, ylabe=None, xy_labe_size=8, run_correlation=True, axis=None, figsize=(8, 4), dpi=150, 
                        precision=2, show_cbar=True, cbar_orient='vertical', cbar_size=1, savefig=False, fig_filename='heatmap.png'):
        """plot heatmap for correlation of dataframe.
        If run_correlation is True, execute df.corr() and plot
        else, plot df"""
        
        cls = MSc_Proj()
        
        if run_correlation:
            corr_vals = np.round(df.corr(), precision)
        else:
            corr_vals = df
            
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.heatmap(corr_vals,  cbar=show_cbar, fmt='0.2f',#f'0.{precision}f',
                               annot_kws={'fontsize':annot_size}, annot=True, cbar_kws={'orientation':cbar_orient,
                                                                                       'shrink':cbar_size})#, square=True,)
        else:
            sns.heatmap(corr_vals, ax=axis, cbar=show_cbar, fmt=f'0.{precision}f',
                       annot_kws={'fontsize':annot_size}, annot=True)#, square=True,)
                         
        axis.set_title(plot_title, size=title_size, weight='bold',)# x=0.5, y=1.05)
        
        axis.set_xticklabels(axis.get_xticklabels(), size=xy_ticklabe_size)
        axis.set_yticklabels(axis.get_yticklabels(), size=xy_ticklabe_size)
        
        if xlabe:
            axis.set_xlabel(xlabe, weight='bold', size=xy_labe_size)
        if ylabe:
            axis.set_ylabel(ylabe, weight='bold', size=xy_labe_size)
        
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis 
    
    @staticmethod
    def plot_pyramid(left_side, right_side, common_catg, left_legend='Left', right_legend='right', xlim=None, ylim=None,
                 plot_title='Pyramid Plot', title_size=15, left_legend_color='white', right_legend_color='white', 
                 x_labe='total_count', y_labe=None, left_labe_shift=0, right_labe_shift=0, rv_labe_shift=0, lv_labe_shift=0,
                 left_side_color='orange', right_side_color='blue', fig_w=6, fig_h=8, savefig=False, fig_filename='pyramid_plot.png', 
                 dpi=200):
        """Pyramid view of negative values vs positive values."""
        
        cls = MSc_Proj()
        negative_side = -left_side
        positive_side = right_side
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        cls.plot_bar(x=negative_side, y=common_catg, 
                   y_labe=y_labe, x_labe=x_labe, bot_labe_color='black',
                   color=left_side_color, annotate=True, v_labe_shift=0.05, axis=ax)

        cls.plot_bar(x=positive_side, y=common_catg, plot_title=plot_title,
                   title_size=title_size, y_labe=y_labe, x_labe=x_labe, bot_labe_color='black',
                   color=right_side_color, annotate=True, v_labe_shift=0.05, axis=ax)

        neg_range = (abs(negative_side.min()) - abs(negative_side.max()))
        pos_range = positive_side.max() - positive_side.min()
        left_pos = negative_side.min() - neg_range/2
        right_pos = positive_side.max() + pos_range/2
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = negative_side.loc[negative_side == negative_side.max()].index[0]
        x_pos = (abs(negative_side.min()) - abs(negative_side.max()))/2
        ax.text(-x_pos-left_labe_shift, min_ind-lv_labe_shift, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':left_side_color})

        labe = right_legend
        max_ind = positive_side.loc[positive_side == positive_side.min()].index[0]
        x_pos = (positive_side.max() - positive_side.min())/2
        ax.text(x_pos+right_labe_shift, max_ind-rv_labe_shift, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':right_side_color})
        
        ax.set_xticklabels(ax.get_xticks())
        plt.show()
        
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
            
    @staticmethod
    def plot_diff(left_side, right_side, common_catgs, left_legend='Left', right_legend='right', xlim=None, ylim=None,
              plot_title='Comparison Plot', title_size=15, left_legend_color='white', right_legend_color='white', 
              precision=2, y_labe=None, x_labe='total_count', left_labe_shift=0, right_labe_shift=0, lv_labe_shift=0,
              rv_labe_shift=0, left_side_color='orange', right_side_color='blue', fig_w=6, fig_h=8, savefig=False, 
              fig_filename='comparison_plot.png', dpi=200):
        """Comparison view of left values vs right values."""
            
        cls = MSc_Proj()
        diff = np.round(right_side - left_side, precision)
        color_mapping = {i: right_side_color if v >= 0 else left_side_color for i, v in zip(common_catgs, diff)}
        #         print(color_mapping)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        cls.plot_bar(x=diff, y=common_catgs, y_labe=y_labe, plot_title=plot_title, title_size=title_size, 
                    x_labe=x_labe, bot_labe_color='black', annotate=True, v_labe_shift=0.05, paletter=color_mapping, 
                    precision=precision, axis=ax)

        left_pos = diff.min() + diff.min()/2
        right_pos = diff.max() + diff.max()/2
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = left_side.loc[left_side == left_side.min()].index[0]
        x_pos = diff.min()/2
        ax.text(x_pos-left_labe_shift, min_ind-lv_labe_shift, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':left_side_color})

        labe = right_legend
        min_ind = right_side.loc[right_side == right_side.min()].index[0]
        x_pos = diff.max()/2
        ax.text(x_pos+right_labe_shift, min_ind-rv_labe_shift, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':right_side_color})
        ax.set_xticklabels(ax.get_xticks())
        plt.show()
        
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
    
            
    @staticmethod
    def split_date_series(date_col, sep='/', year_first=True):
        """split series containing date data in str format into
        dataframe of tdayee columns (year, month, day)
        Return:
        Dataframe"""
        
        date_col = date_col.str.split(sep, expand=True)
        if year_first:
            day = date_col[2]
            mon = date_col[1]
            yr = date_col[0]
        else:
            day = date_col[0]
            mon = date_col[1]
            yr = date_col[2]
        
        return pd.DataFrame({'Year': yr,
                            'Month': mon,
                            'Day': day})
        
    @staticmethod
    def file_search(search_from: 'path_like_str'=None, search_pattern_in_name: str=None, search_file_type: str=None, print_result: bool=False):
        """
        returns a str containing the full path/location of all the file(s)
        matching the given search pattern and file type
        """
    
        # raise error when invalid arguments are given
        if (search_from is None):
            raise ValueError('Please enter a valid search path')
        if (search_pattern_in_name is None) and (search_file_type is None):
            raise ValueError('Please enter a valid search pattern and/or file type')
        
        search_result = {}
        print(f"Starting search from: {search_from}\n")
        for fpath, folders, files in os.walk(search_from):
            for file in files:
                # when both search pattern and file type are entered
                if (search_file_type is not None) and (search_pattern_in_name is not None):
                    if (search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]) and \
                            (search_pattern_in_name.lower() in file.lower().split('.')[0]):
                        search_result.setdefault(file, f'{fpath}\\{file}')

                # when file type is entered without any search pattern
                elif (search_pattern_in_name is None) and (search_file_type is not None):
                    # print(search_file_type)
                    if search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]:
                        search_result.setdefault(file, f'{fpath}\\{file}')    

                # when search pattern is entered without any file type
                elif (search_file_type is None) and (search_pattern_in_name is not None):
                    if search_pattern_in_name.lower() in file.lower().split('.')[0]:
                        search_result.setdefault(file, f'{fpath}\\{file}')
                        
        if print_result:
            for k,v in search_result.items():
                print(f"{k.split('.')[0]} is at {v}")
                
        return search_result
        
    @staticmethod
    def visualize_nulls(df, plot_title='Missing Entries per Variable', annotate=True, annot_size=6, use_bar=False, 
                        include_perc=True, perc_total=None, perc_labe_gap=0.01, h_labe_shift=0.1, perc_labe_color='black',
                        color=None, reduce_barw_by=1, fig_size=(8, 6), dpi=200, savefig=False, fig_filename='missing_data.png'):
                """plot count plot for null values in df."""
                
                cls = MSc_Proj()
                
                null_cols = cls.null_checker(df, only_nulls=True)
                y_range = null_cols.max() - null_cols.min()
                
                if include_perc:
                    perc_nulls = cls.null_checker(df, only_nulls=True, in_perc=True)
                
                
                if not len(null_cols):
                    return 'No null values in the dataframe'

                sns.set_style("whitegrid")
                
                fig = plt.figure(figsize=fig_size, dpi=dpi)
                ax = fig.add_axes([0, 0, 1, 1])
                
                if not color:
                    color = 'brown'
                    
                if use_bar:  #use bar chart
                    cls.plot_bar(null_cols, null_cols.index, plot_title=plot_title, rotate_yticklabe=False,
                                  y_labe='Column Names', x_labe='Number of Missing Values', include_perc=include_perc,
                                  perc_total=perc_total, annotate=annotate, annot_size=annot_size, perc_labe_gap=perc_labe_gap, 
                                  v_labe_shift=h_labe_shift, perc_labe_color=perc_labe_color, color=color, reduce_barw_by=reduce_barw_by, 
                                  figsize=fig_size, dpi=dpi, axis=ax)
                    
                    plt.xlim(right=null_cols.max()+y_range/2)
                else:  #use column chart
                    cls.plot_column(x=null_cols.index, y=null_cols, plot_title=plot_title, rotate_xticklabe=True, 
                                    x_labe='Column Names', y_labe='Number of Missing Values', include_perc=include_perc, 
                                    annotate=annotate, annot_size=annot_size, perc_labe_gap=perc_labe_gap, h_labe_shift=h_labe_shift,
                                    color=color, perc_labe_color=perc_labe_color, reduce_barw_by=reduce_barw_by, 
                                    figsize=fig_size, dpi=dpi, axis=ax)
                    
                    plt.ylim(top=null_cols.max()+y_range/2)
                plt.show()
                if savefig:
                    print(cls.fig_writer(fig_filename, fig))
    
    @staticmethod
    def null_checker(df: pd.DataFrame, in_perc: bool=False, only_nulls=False):
        """return quantity of missing data per dataframe column."""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")
            
        null_df = df.isnull().sum().sort_values(ascending=True)
            
        if in_perc:
            null_df = (null_df*100)/len(df)
            if len(null_df):
                print("\nIn percentage")
        
        if only_nulls:
            null_df = null_df.loc[null_df > 0]
            if len(null_df):
                print("\nOnly columns with null values are included")
            
        return np.round(null_df, 2)
        
        
    @staticmethod
    def check_for_empty_str(df: pd.DataFrame):
        """Return True for column containing '' or ' '.
        Output is a dictionary with column name as key and True/False as value."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")

        cols = list(df.columns)  # list of columns
        result = dict()
        for i in range(len(cols)):
            # True for columns having empty strings
            result[cols[i]] = df.loc[(df[cols[i]] == ' ') |
                                    (df[cols[i]] == '')].shape[0] > 0
        
        result = pd.Series(result)
        
        return result.loc[result == True]
    
    @staticmethod
    def col_blank_rows(df: pd.DataFrame):
        """check an entire dataframe and return dict with columns containing blank rows 
        (as keys) and a list of index of blank rows (values)."""
      
        cls = MSc_Proj()
        blank_cols = cls.check_for_empty_str(df).index
        #blank_cols = guide.loc[guide[col for col, is_blank in guide.items() if is_blank]
        result = dict()
        for i in range(len(blank_cols)):
            result[blank_cols[i]] = list(df[blank_cols[i]].loc[(df[blank_cols[i]] == " ") |
                                                          (df[blank_cols[i]] == "")].index)
        return result
    
    @staticmethod
    def fig_writer(fname: str, plotter: plt.figure=None, dpi: int=200, file_type='png'):
        """save an image of the given figure plot in the filesystem."""
        
        cls = MSc_Proj()
        plotter.get_figure().savefig(f"{cls.app_folder_loc}\\{fname}", dpi=dpi, format=file_type,
                                         bbox_inches='tight', pad_inches=0.25)
        return fname