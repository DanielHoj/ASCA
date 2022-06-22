import pandas as pd
import numpy as np
import itertools
import patsy.contrasts as cont
import statsmodels.api as sm
import scipy.stats as st
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from matplotlib.lines import Line2D


@dataclass
class ASCA_Results:
    residuals = None
    sca_results = None
    beta: np.ndarray = None
    mu_n: list = None
    mu: np.ndarray = None
    column_names: list = None
    factor_names: list = None
    factors: int = None
    dummy: np.ndarray = None
    dummy_n: list = None
    dummy_indexer: list = None
    p_value: list = None
    permutation_sse: np.ndarray = None
    contrast_matrix: np.ndarray = None
        
class ASCA_Options:
    def __init__(self, n_components, interaction, n_perm):
        self.n_components = n_components
        self.interaction = interaction
        self.n_perm = n_perm
        
    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, n_components):
        if not isinstance(n_components, int):
            raise ValueError('Options n_components: Please input integer between 2 and 15')
        if not 2 <= n_components <= 15:
            raise ValueError('Options n_components: Value needs to be between 2 and 15')
        self._n_components = n_components
        
    @property
    def n_perm(self):
        return self._n_perm

    @n_perm.setter
    def n_perm(self, n_perm):
        if not isinstance(n_perm, int):
            raise ValueError('Options n_perm: Please input integer between 2 and 15')
        if not 2 < n_perm:
            raise ValueError('Options n_perm: Value needs to be more than 1')
        self._n_perm = n_perm

    @property
    def interaction(self):
        return self._interaction
    
    @interaction.setter
    def interaction(self, interaction):
    # Assert interaction is a boolean
        if not isinstance(interaction, bool):
            raise ValueError('Options interaction: Please input boolean True/False')
        self._interaction = interaction
        
class ASCA:  
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._x = None
        self._y = None
        self.Options = ASCA_Options(5, False, 1000)
        self._results = ASCA_Results()
    
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, dataframe):
        if not isinstance(dataframe, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError('X: Please input pandas DataFrame, Series or numpy ndarray')
        self._X = dataframe
        
    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, dataframe):
        if not isinstance(dataframe, (pd.DataFrame, np.ndarray)):
            raise ValueError('Y: Please input pandas DataFrame or numpy ndarray')
        self._Y = dataframe
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Methods
    '''
        
    def fit(self, X_is_dummy = False):
        
        self.__checkX()
        self.__checkY()
        
        if X_is_dummy == True:
            self._results.dummy = self._x
            
        elif X_is_dummy == False:
            self.__dummy_function(self._x)
        
        self.__partition_matrix()
        self.__sca()
        

    def plot_scores(self, factor = 0, group_by = None, components = (0, 1)):
        
        # if factor is a string (factor name) assert it is in factor_names and
        # translate to its index
        if isinstance(factor, str):
            try:
                factor_number = self._results.factor_names.index(factor)
            except:
                raise ValueError('The factor is not in factor_names')
        elif isinstance(factor, int):
            factor_number = factor
        else:
            raise ValueError('Factor input needs to be a string or an int')
        
        # Get the SCA results and raise warning if .fit() has not been called
        sca_results = self._results.sca_results
        
        if sca_results is None:
            raise AttributeError('Please call fit() first')
        
        # Allow group_by to be a string of the column name of x
        if isinstance(group_by, str):
            try:
                group_by = getattr(self.X, group_by)
            except:
                   raise ValueError(f'{group_by} not a factor name')
                   
        
        # Automatically assign the group_by to factor name if none is assigned
        if group_by is None:
            if self._results.factors == 1:
                group_by = pd.Series(self._x, name = self._results.factor_names[0])
            else:
                # if .Options.interactions = True, then factor_number is changed
                try:
                    X = pd.DataFrame(self._x, columns = self._results.factor_names)
                    group_by = self.X.iloc[:,factor_number]
                except:
                    if len(self._x.shape) == 1:
                        diff = self._results.factors - 1
                    else:
                        diff = self._results.factors - self._x.shape[1]
                        
                    group_by = self.X.iloc[:,factor_number-diff]
                    
        self.__plot_sca(sca_results[factor_number], d = group_by, components = components, factor_p = factor_number)
        
    def plot_loadings(self, factor = 0, components = (0, 1)):
        # if factor is a string (factor name) assert it is in factor_names and
        # translate to its index
        if isinstance(factor, str):
            try:
                factor_number = self._results.factor_names.index(factor)
            except:
                raise ValueError('The factor is not in factor_names')
                
        elif isinstance(factor, int):
            factor_number = factor
        else:
            raise ValueError('Factor input needs to be a string or an int')
        
        #  Get the SCA results and raise warning if .fit() has not been called
        sca_results = self._results.sca_results
        if sca_results is None:
            raise AttributeError('Please call fit() first')
            
        self.__plot_sca(sca_results[factor_number], plot_type = 'loadings', components = components, factor_p = factor_number)
        
    # Plot raw data
    def plot_raw(self, factor = None):
        
        # Assert fit() has been called
        if self._results.factor_names is None:
            raise ValueError('Please call fit() first')
        
        # make Y as pandas DataFrame
        Y  = pd.DataFrame(self._y, columns= self._results.column_names)
        
        # If no factor to colour from
        if factor is None:
            Y.T.plot(legend = False)
        
        # Else color based on factor
        else:
            # Check if factor is string of int
            if isinstance(factor, str):
                try:
                    factor_number = self._results.factor_names.index(factor)
                    factor_name = factor
                except:
                    raise ValueError(f'{factor} not in factor_names')
            elif isinstance(factor, int):
                factor_number = factor
                factor_name = self._results.factor_names[factor]
            else:
                raise ValueError('Factor input needs to be a string or an int')
                
            if len(self._x.shape) == 1:
                factor = pd.Series(self._x, name = factor_name)
            else:
                factor = pd.Series(self._x[:,factor_number], name = factor_name)
            
            result = pd.concat([factor, Y], axis = 1)
            
            unique = factor.nunique()
            
            if unique <21:
                colors = self.__distinct_colors(unique)
            else:
                colors = mcp.gen_color(cmap ='prism', n=unique)
                
            if factor.dtype != 'int':
                codes, unique = pd.factorize(factor)
                result['codes'] = codes
                result['colors'] = result.loc[:,f'{factor.name}'].map({name:colors[codes[i]]
                                     for i, name in enumerate(result.loc[:,f'{factor.name}'])})
            else:
                result['colors'] = result.loc[:,f'{factor.name}'].map({i:colors[i-1]
                                             for i in factor})
                
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label=f'{factor.name}: {factor.unique()[i]}', markerfacecolor=mcolor, markersize=5)
                        for i, mcolor in enumerate(colors)]
            
            Y.T.plot(legend = False, color = result.colors)
            plt.legend(handles = legend_elements, ncol = 2)
            plt.title(f'Raw data for {factor_name}', loc = 'left')
    
    def plot_residual(self, factor = None):
        
        # Assert fit() has been called
        if self._results.factor_names is None:
            raise ValueError('Please call fit() first')
        
        # make residuals as pandas DataFrame
        residual  = pd.DataFrame(self._results.residuals, columns= self._results.column_names)
        
        # If no factor to colour from
        if factor is None:
            residual.T.plot(legend = False)
            plt.title(f'Residuals', loc = 'left')
        
        # Else color based on factor
        else:
            # Check if factor is string of int
            if isinstance(factor, str):
                try:
                    factor_number = self._results.factor_names.index(factor)
                    factor_name = factor
                except:
                    raise ValueError('The factor is not in factor_names')
            elif isinstance(factor, int):
                factor_number = factor
                factor_name = self._results.factor_names[factor]
            else:
                raise ValueError('Factor input needs to be a string or an int')
            
            if len(self._x.shape) == 1:
                factor = pd.Series(self._x, name = factor_name)
            else:
                factor = pd.Series(self._x[:,factor_number], name = factor_name)
            
            result = pd.concat([factor, residual], axis = 1)
            
            unique = factor.nunique()
            
            if unique <21:
                colors = self.__distinct_colors(unique)
            else:
                colors = mcp.gen_color(cmap ='prism', n=unique)
                
            if factor.dtype != 'int':
                codes, unique = pd.factorize(factor)
                result['codes'] = codes
                result['colors'] = result.loc[:,f'{factor.name}'].map({name:colors[codes[i]]
                                     for i, name in enumerate(result.loc[:,f'{factor.name}'])})
            else:
                result['colors'] = result.loc[:,f'{factor.name}'].map({i:colors[i-1]
                                             for i in factor})
                
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label=f'{factor.name}: {factor.unique()[i]}', markerfacecolor=mcolor, markersize=5)
                        for i, mcolor in enumerate(colors)]
            
            residual.T.plot(legend = False, color = result.colors)
            plt.legend(handles = legend_elements, ncol = 2)
            plt.title(f'Residuals coloured by {factor_name}', loc = 'left')


    def plot_permutations(self, bins = None):
        
        # Assert .permutation_test() has been called
        check = getattr(self._results, 'p_value')
        if check is None:
            raise ValueError('Please run permutation_test() first')
        
        # Set bins to default
        if bins is None:
            bins = int(self.Options.n_perm/2)
            if bins > 100:
                bins = 100
        
        # Get SSE as pandas dataframe
        df_sse = pd.DataFrame(self._results.permutation_sse, columns = self._results.factor_names)
        
        # Plot histogram of SSE
        ax = df_sse.plot.hist(bins = bins)
        plt.xlabel('SSE')
        
        # Insert v line at SSE of the full model
        plt.axvline(np.sum(np.square(self._results.residuals)), ls = '--', color = 'r')
        ax.legend(bbox_to_anchor = (1.0,1), loc = 'upper left')
        
        # Set limits of y and x
        if np.sum(np.square(self._results.residuals)) > df_sse.values.min():
            minlim = df_sse.values.min() - (df_sse.values.max() - df_sse.values.min())*.1
            maxlim = df_sse.values.max() + (df_sse.values.max() - df_sse.values.min())*.1
        else:
            minlim = np.sum(np.square(self._results.residuals)) - (df_sse.values.max() - np.sum(np.square(self._results.residuals)))*.1
            maxlim = df_sse.values.max() + (df_sse.values.max() - df_sse.values.min())*.1
        plt.xlim(minlim, maxlim)
        
        # Write 'SSE Full model'
        y_mean = ax.get_ylim()[1]/1.2
        ax.text((maxlim-minlim)*.01 + np.sum(np.square(self._results.residuals)),y_mean, 'SSE \nFull model', size = 15)
    
    def permutation_test(self):
        
        #
        self.fit()
        
        contrast = self._results.contrast_matrix
        Y = self._y
        X = self._x
        factors = self._results.factors
        
        sse = np.asarray(np.repeat(np.sum(np.square(self._results.residuals)), factors))
        
        ssep = np.asarray([self.__permuter(Y, contrast, factors) 
           for i in range(self.Options.n_perm) if print(f'Permutation {i+1} of {self.Options.n_perm}')
           or True])
        
        booli = (ssep < sse)
    
        # p-value number of ssep < sse_m/n_perm
        p_value = np.asarray([(booli[:,i].sum()+1)/(self.Options.n_perm+1)
                  for i in range(factors)])
        
        self._results.permutation_sse = ssep
        self._results.p_value = p_value
        
        return p_value
        
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods
    Permutation
    '''
    


    def __permuter(self, Y, contrast, factors):
       
        
        dummy_shuffled = self.__shuffled()
        
        mu_n = [self.__partition_matrix_permute(Y, dummy)
                 for dummy in dummy_shuffled]
        
        ssep = np.asarray([np.sum(np.square(Y - mu))
                for mu in mu_n])
                          
        return ssep

    def __shuffled(self):
        
        # permutate all dummy_n 
        X = self._results.dummy_n
        x = X[1:].copy()
        

        shuffled = [np.random.permutation(xi) 
                        for xi in x]
        
        result = []
        for i in range(len(shuffled)):
            dummy = X.copy()
            dummy[i+1] = shuffled[i]
            result.append(np.asarray(np.concatenate(dummy, axis =1)))
            
        return result
    
    def __partition_matrix_permute(self, Y, dummy):   
        
        beta = np.linalg.inv(dummy.T@dummy)@dummy.T@Y
        
        #Calculate X_hat for the full model:
        mu = dummy@beta
        
        return mu
    
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods
    Input checks
    '''
    
    
    
    def __checkX(self):
        X = self.X
        if isinstance(X, pd.DataFrame):
            self._results.factor_names = list(X.columns)
            self._results.factors = X.shape[1]

            X = X.values
            self.X = pd.DataFrame(X, columns = self._results.factor_names)
            
            if len(X.shape) > 1 and X.shape[1] == 1:
                X = X.reshape(X.shape[0],)
            
        elif isinstance(X, pd.Series):
            self._results.factors = 1
            self._results.factor_names = [X.name]
            
            X = X.values
            self.X = pd.DataFrame(X, columns = self._results.factor_names)
            
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                self._results.factors = 1
                self._results.factor_names = ['Factor 1']
                
                self.X = pd.Series(X, name = 'Factor 1')
                
            else:
                self._results.factors = X.shape[1]
                self._results.factor_names = [f'Factor {k+1}' 
                                          for k in range(X.shape[1])]
                
                self.X = pd.DataFrame(X, columns = self._results.factor_names)
                
        #self.X = pd.DataFrame(X, columns = self._results.factor_names)
        self._x = X
        
    def __checkY(self):
        Y = self.Y
        if isinstance(Y, pd.DataFrame):
            self._results.column_names = list(Y.columns)
            Y = Y.values
            
        elif isinstance(Y, np.ndarray):
            self._results.column_names = [f'Variable {k+1}' 
                                          for k in range(Y.shape[1])]
        self.Y = pd.DataFrame(Y, columns = self._results.column_names)
        self._y = Y
        
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods
    Matrix partition
    '''
    
    def __partition_matrix(self):
    
        # Get dummy matrix
        dummy = self._results.dummy
        
        # Get list of individual dummys
        dummy_n = self._results.dummy_n
        
        # Set an indexer based on dummys
        self._results.dummy_indexer = self.__dummy_index(dummy_n)
        
        # Get indexer based on dummys
        dummy_indexer = self.__dummy_index(dummy_n)
        
        # Drop intercept
        dummy_n = dummy_n[1:]
        
        # Calculate regression coeficients, beta
        self._results.beta = np.linalg.inv(dummy.T@dummy)@dummy.T@self._y
        
        
        # Subset beta for the reduced model
        self._results.beta_n = [self._results.beta[dummy_indexer[i],:] 
                       for i in range(1, self._results.factors + 1)]
        
        # Calculate mu for the reduced model
        self._results.mu_n = [dummy_n[i]@self._results.beta_n[i]
                  for i in range(self._results.factors)]
        
        # Calculate mu for the full model
        self._results.mu = dummy @ self._results.beta
        
        # Calculate residuals
        self._results.residuals = self._y - self._results.mu

    def __dummy_function(self, X):
        # Factorize X
        if X.dtype != 'int':
            X = self.__factorize(X)
        
        # Prepare X
        if len(X.shape) > 1:
            rows, cols = X.shape
            X = list(X.transpose())
            n_factors = len(X)
        else:
            X = X.transpose()
            n_factors = 1
            rows = X.shape[0]
            
        #Make contrast matrix
        contrast = cont.Sum()
        intercept = np.asmatrix(np.ones(rows)).transpose()
        
        #for one factor:
        if n_factors == 1:
            
            #number of levels:
            levels = list(set(X))
            C = contrast.code_with_intercept(levels)
            
            result = C.matrix[X-1,:]
            self._results.dummy_n = [intercept] + [result[:,1:]]
        
        # for multiple factors
        else:            
            # number of levels for each factor
            levels = [list(set(X_i)) 
                      for X_i in X]
            
            # Contrast matrices
            C = [contrast.code_without_intercept(factor).matrix 
                 for factor in levels]
            
            # temp (dummy_n)
            temp = [intercept] + [C[i][X[i]-1,:] 
                    for i in range(len(C))]
            self._results.dummy_n = temp
            
            result = np.concatenate(temp, axis =1)
            
            # if Options.interaction = True
            if self.Options.interaction:
                combinations = list(itertools.combinations(self._results.factor_names,2))
                extra_names = ['*'.join(i) for i in combinations]
                extras = [a*b for a,b in itertools.combinations(temp[1:], 2)]
                
                temp_extras = temp + extras
                result = np.concatenate(temp_extras, axis =1)
                
                # Assert design rank i not higher than Y
                if result.shape[1] > self._y.shape[1]-1:
                    self.Options.interaction = False
                    raise ValueError('Design rank higher than Y rank, interaction set to False')
                self._results.factor_names += extra_names
                self._results.factors += len(extra_names)
                self._results.dummy_n = temp_extras
        
        # Set dummy matrix
        self._results.dummy = np.asarray(result)
        self._results.contrast_matrix = C
        
    def __dummy_index(self, x):
        # Create indexer based on dummy_n
        xc = [0, *np.cumsum([i.shape[1] for i in x])]
        y = [np.asarray([q >= xc[n] and q < xc[n+1] for q in range(xc[-1])]) 
             for n in range(len(x))]
        return y
    
    def __factorize(self, X):
        # Factorize X if are strings
        if len(X.shape) == 1:
            newX = np.unique(X, return_inverse=True)[1]
        else:
            newX = np.asarray([np.unique(X[:,i], return_inverse=True)[1] 
                    for i in range(X.shape[1])]).T
        return newX
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods
    SCA
    '''
    
    def __sca(self):
        residuals = self._results.residuals
        
        result = []
        
        for k in range(self._results.factors):
            #List of name of pcs:
            PCn = [f'PC{num}' for num in range(1, self.Options.n_components+1)]
            #Covariance matrix of X
            cov_mat = np.cov(self._results.mu_n[k] , rowvar = False)
             
            #calculate eigenValues and eigen vectors
            eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
             
            #Sort based on eigenvalues
            sorted_index = np.argsort(eigen_values)[::-1][:self.Options.n_components]
            sorted_eigenvalues = eigen_values[sorted_index]
            sorted_eigenvectors = eigen_vectors[:,sorted_index]
            
            #Keep n_components
            eigenvector_subset = sorted_eigenvectors[:,0:self.Options.n_components]
            
            Loadings = pd.DataFrame(eigenvector_subset*np.sqrt(sorted_eigenvalues), 
                       index=self._results.column_names, columns = PCn).T
        
            Scores = pd.DataFrame(np.dot((self._results.mu_n[k]+residuals),eigenvector_subset), columns=PCn)
            
            explained_variance = sorted_eigenvalues/np.sum(eigen_values)
        
            result.append([Scores, Loadings, explained_variance, self._results.factor_names[k]])
        
        self._results.sca_results = result
    
    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Private methods
    Plots
    '''

    def __plot_sca(self, sca_results, d = None, components = (0,1), legend = 'on', 
                plot_type ='scores', factor_p = None):
        
        # Get variables for speficic SCA result
        scores, loadings, explained_variance, factor_names = sca_results
        
        name = ' of ' + factor_names
        
        if factor_p is not None:
            p = getattr(self._results, 'p_value')
            if p is not None:
                p_title = '\np: ' + str(p[factor_p].round(3))
            else:
                p_title = ''
        else:
            p_title = ''
            
        if plot_type == 'scores':
            scores = scores.iloc[:,[components[0],components[1]]]
            if isinstance(d, (pd.Series, np.ndarray)):
                # Create spider plot
                D_scores = pd.concat([d, scores], axis = 1)
                Centerpoints = D_scores.groupby(f'{d.name}').mean()
                D_scores = D_scores.set_index(f'{d.name}')
                D_scores.loc[:,f'Center_{D_scores.columns[0]}'] = Centerpoints.iloc[:,0]
                D_scores.loc[:,f'Center_{D_scores.columns[1]}'] = Centerpoints.iloc[:,1]
                D_scores.loc[:,f'{d.name}'] = D_scores.index
                if d.nunique() <21:
                    colors = self.__distinct_colors(d.nunique())
                else:
                    colors = mcp.gen_color(cmap ='prism', n=d.nunique())
                
                if d.dtype != 'int':
                    codes, unique = pd.factorize(d)
                    D_scores['codes'] = codes
                    D_scores['colors'] = D_scores.loc[:,f'{d.name}'].map({name:colors[codes[i]]
                                         for i, name in enumerate(D_scores.loc[:,f'{d.name}'])})
                else:
                    D_scores['colors'] = D_scores.loc[:,f'{d.name}'].map({i:colors[i-1]
                                         for i in d})
                
                fig, ax = plt.subplots(1, figsize=(8,8))
            
                plt.scatter(scores.iloc[:,0], scores.iloc[:,1], c=D_scores.colors, alpha = 0.6, s=10)
                # plot lines
                for idx, val in D_scores.iterrows():
                    x = [val[f'{scores.columns[0]}'], val[f'Center_{scores.columns[0]}']]
                    y = [val[f'{scores.columns[1]}'], val[f'Center_{scores.columns[1]}']]
                    plt.plot(x, y, c = val.colors, alpha=0.4)
                
                if legend == 'on':
                    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                    label=f'{d.name}: {d.unique()[i]}', markerfacecolor=mcolor, markersize=5) 
                    for i, mcolor in enumerate(colors)]
                    plt.legend(handles=legend_elements, loc='upper right', ncol=2)
            
    
                plt.xlim(D_scores.iloc[:,0].min() -(D_scores.iloc[:,0].max() -D_scores.iloc[:,0].min())*.05,D_scores.iloc[:,0].max() +(D_scores.iloc[:,0].max() -D_scores.iloc[:,0].min())*.05)
                plt.ylim(D_scores.iloc[:,1].min() -(D_scores.iloc[:,1].max() -D_scores.iloc[:,1].min())*.05,D_scores.iloc[:,1].max() +(D_scores.iloc[:,1].max() -D_scores.iloc[:,1].min())*.05)
                
                plt.axhline(linestyle='--')
                plt.axvline(linestyle='--')    
                
                plt.title(f'Scores{name} {p_title}', loc='left', fontsize=22)
                plt.xlabel(f'{scores.columns[0]} \n Explained Variance: {round(explained_variance[components[0]]*100, 1)}%')
                plt.ylabel(f'{scores.columns[1]} \n Explained Variance: {round(explained_variance[components[1]]*100, 1)}%')
            
            else:
                D_scores = scores
                fig, ax = plt.subplots(1, figsize=(8,8))      
                plt.scatter(scores.iloc[:,0], scores.iloc[:,1], alpha = 0.6, s=10)
                
                plt.axhline(linestyle='--')
                plt.axvline(linestyle='--')    
                
                plt.title(f'scores{name} {p_title}', loc='left', fontsize=22)
                plt.xlabel(f'{scores.columns[0]} \n Explained Variance: {round(explained_variance[components[0]]*100, 1)}%')
                plt.ylabel(f'{scores.columns[1]} \n Explained Variance: {round(explained_variance[components[1]]*100, 1)}%')
            
                plt.xlim(scores.iloc[:,0].min() -(scores.iloc[:,0].max() -scores.iloc[:,0].min())*.05,scores.iloc[:,0].max() +(scores.iloc[:,0].max() -scores.iloc[:,0].min())*.05)
                plt.ylim(scores.iloc[:,1].min() -(scores.iloc[:,1].max() -scores.iloc[:,1].min())*.05,scores.iloc[:,1].max() +(scores.iloc[:,1].max() -scores.iloc[:,1].min())*.05)
        
        elif plot_type == 'loadings':
            title = f'Loadings{name} {p_title}'
            xlabel = 'Variable'
            loadings.T.iloc[:,list(components)].plot(title = title, ylabel = xlabel)

        
        
    def __distinct_colors(self, num_colors):
        # most distinct colors for 2-20
        #max colors = 20
        colors= [
        ['#00ff00', '#0000ff'],
        ['#ff0000', '#00ff00', '#0000ff'],
        ['#ff0000', '#00ff00', '#0000ff', '#87cefa'],
        ['#ffa500', '#00ff7f', '#00bfff', '#0000ff', '#ff1493'],
        ['#66cdaa', '#ffa500', '#00ff00', '#0000ff', '#1e90ff', '#ff1493'],
        ['#808000', '#ff4500', '#c71585', '#00ff00', '#00ffff', '#0000ff', 
         '#1e90ff'],
        ['#006400', '#ff0000', '#ffd700', '#c71585', '#00ff00', '#00ffff', 
         '#0000ff', '#1e90ff'],
        ['#191970', '#006400', '#bc8f8f', '#ff4500', '#ffd700', '#00ff00', 
         '#00ffff', '#0000ff', '#ff1493'],
        ['#006400', '#00008b', '#b03060', '#ff4500', '#ffff00', '#deb887', 
         '#00ff00', '#00ffff', '#ff00ff', '#6495ed'],
        ['#8b4513', '#006400', '#4682b4', '#00008b', '#ff0000', '#ffff00', 
         '#00ff7f', '#00ffff', '#ff00ff', '#eee8aa', '#ff69b4'],
        ['#2f4f4f', '#7f0000', '#008000', '#000080', '#ff8c00', '#ffff00',
         '#00ff00', '#00ffff', '#ff00ff', '#1e90ff', '#eee8aa', '#ff69b4'],
        ['#2f4f4f', '#8b4513', '#228b22', '#000080', '#ff0000', '#ffff00',
         '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#1e90ff', '#eee8aa', 
         '#ff69b4'],
        ['#2f4f4f', '#7f0000', '#008000', '#4b0082', '#ff8c00', '#deb887', 
         '#00ff00', '#00bfff', '#0000ff', '#ff00ff', '#ffff54', '#dda0dd', 
         '#ff1493', '#7fffd4'],
        ['#2f4f4f', '#8b4513', '#006400', '#4b0082', '#ff0000', '#ffa500', 
         '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#d8bfd8', '#ff00ff', 
         '#1e90ff', '#98fb98', '#ff69b4'],
        ['#2f4f4f', '#800000', '#191970', '#006400', '#bdb76b', '#48d1cc', 
         '#ff0000', '#ffa500', '#ffff00', '#0000cd', '#00ff00', '#00fa9a', 
         '#da70d6', '#d8bfd8', '#ff00ff', '#1e90ff'],
        ['#2f4f4f', '#800000', '#008000', '#bdb76b', '#4b0082', '#b03060', 
         '#48d1cc', '#ff4500', '#ffa500', '#ffff00', '#00ff00', '#00fa9a', 
         '#0000ff', '#d8bfd8', '#ff00ff', '#1e90ff', '#ee82ee'],
        ['#2f4f4f', '#7f0000', '#006400', '#7f007f', '#ff0000', '#ff8c00',
         '#ffff00', '#40e0d0', '#7fff00', '#00fa9a', '#4169e1', '#e9967a', 
         '#00bfff', '#0000ff', '#ff00ff', '#f0e68c', '#dda0dd', '#ff1493'],
        ['#808080', '#2e8b57', '#7f0000', '#808000', '#8b008b', '#ff0000',
         '#ffa500', '#ffff00', '#0000cd', '#7cfc00', '#00fa9a', '#4169e1', 
         '#00ffff', '#00bfff', '#f08080', '#ff00ff', '#eee8aa', '#dda0dd',
         '#ff1493'],
        ['#2f4f4f', '#2e8b57', '#8b0000', '#808000', '#7f007f', '#ff0000',
         '#ff8c00', '#ffd700', '#0000cd', '#00ff7f', '#4169e1', '#00ffff', 
         '#00bfff', '#adff2f', '#d8bfd8', '#ff00ff', '#f0e68c', '#fa8072',
         '#ff1493', '#ee82ee']
        ]
        try:
            color = colors[num_colors-2]
            return color
        except:
            print('Too many colors')
            
    
    
