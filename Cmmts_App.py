import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import graphviz
import xgboost as xgb

# CSS to inject contained in a string
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 16px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
# Streamlit app
def main():
    st.title("Non-Linear Driver Analysis App")

    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and random forest classification. 
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Upload and filter datasets
            - Perform factor analysis with customizable settings
            - Train and evaluate a Random Forest classifier with optional hyperparameter tuning
            - Visualize results with ROC curves and feature importance
                
            ---
            """, unsafe_allow_html=True)
    
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Show original data shape
        st.write("Original Data Shape:")
        st.write(df.shape)

        # Multiple filtering options
        st.header("Filter Data")
        filter_columns = st.multiselect("Select columns to filter:", df.columns)
        filters = {}
        for col in filter_columns:
            unique_values = df[col].unique()
            if pd.api.types.is_numeric_dtype(df[col]):
                selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                filters[col] = selected_values
            else:
                selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                filters[col] = selected_values

        filtered_df = df.copy()
        for col, selected_values in filters.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

        st.write("Filtered Data:")
        st.write(filtered_df)

        # Show filtered data shape
        st.write("Filtered Data Shape:")
        st.write(filtered_df.shape)

        # Allow user to select the outcome and independent variables
        outcome_var = st.selectbox("Select the outcome variable:", filtered_df.columns)
        independent_vars = st.multiselect("Select independent variables:", filtered_df.columns)
        
        if outcome_var and independent_vars:
            df2 = filtered_df[independent_vars]
            y = filtered_df[outcome_var]

            # Perform statistical tests and plots
            st.header("Statistical Tests and Plots")

            # Bartlett’s Test of Sphericity
            chi2, p = calculate_bartlett_sphericity(df2)
            st.markdown("**Bartlett’s Test of Sphericity:**")
            st.write(f"Chi-squared value: {chi2}, p-value: {p:.3f}")
            with st.expander("Description"):
                        st.write("""
                        **What it is**: A statistical test used to examine the hypothesis that the variables in a dataset are uncorrelated.
            
                        **What it tells us**: If the test is significant (p < 0.05), it indicates that the variables are correlated and suitable for factor analysis.
                        """)


            # Kaiser-Meyer-Olkin (KMO) Test
            kmo_values, kmo_model = calculate_kmo(df2)
            st.write("**Kaiser-Meyer-Olkin (KMO) Test:**")
            st.write(f"KMO Test Statistic: {kmo_model}")
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: This test measures the adequacy of sampling for factor analysis.
                        
                        **What it tells us**: A KMO value closer to 1 indicates that the data is suitable for factor analysis. Values below 0.6 generally indicate the data is not suitable for factor analysis.
                        """)

            # Scree Plot
            fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=df2.shape[1])
            fa.fit(df2)
            ev, _ = fa.get_eigenvalues()
            plt.figure(figsize=(10, 6))
            plt.scatter(range(1, df2.shape[1] + 1), ev)
            plt.plot(range(1, df2.shape[1] + 1), ev)
            plt.title('Scree Plot')
            plt.xlabel('Factors')
            plt.ylabel('Eigen Value')
            plt.grid()
            st.pyplot(plt)
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: A graph showing the eigenvalues of the factors in descending order.
                        
                        **What it tells us**: Helps to determine the number of factors to retain by identifying the point where the curve starts to flatten (the "elbow").
                        """)

            # Heatmap of correlation matrix
            plt.figure(figsize=(20, 10))
            sns.heatmap(df2.corr(), cmap="Reds", annot=True)
            st.pyplot(plt)
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: A visual representation of the correlation matrix where the strength of correlation is represented by color intensity.
                        
                        **What it tells us**: Helps to identify the strength and direction of relationships between variables. High correlation values indicate multicollinearity.
                        """)

            # Variance Inflation Factor (VIF)
            df2_with_const = add_constant(df2)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = df2_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(df2_with_const.values, i) for i in range(df2_with_const.shape[1])]
            vif_data = vif_data[vif_data["Variable"] !="const"]
            st.write("Variance Inflation Factor (VIF):")
            st.write(vif_data)
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: Measures the increase in variance of the estimated regression coefficients due to collinearity.
                        
                        **What it tells us**: VIF values above 10 indicate high multicollinearity, suggesting that the predictor variables are highly correlated and may not be suitable for regression analysis.
                        """)

            # Factor Analysis
            st.subheader("Factor Analysis")

            # if st.checkbox("Click to select method and rotation"):
            #     rotation_options = ["None", "Varimax", "Promax", "Quartimax", "Oblimin"]
            #     rotation = st.selectbox("Select rotation:", rotation_options)
            #     method_options = ["Principal", "Minres", "ML", "GLS", "OLS"]
            #     method = st.selectbox("Select method:", method_options)
            #     if rotation == "None":
            #         rotation = None
            #     if method == "Principal":
            #         method = "principal"
            # else:
            rotation = "varimax"
            method = "principal"

            st.write(f"Method: {method}, Rotation: {rotation}")
            # Explanation for Principal and Varimax
            with st.expander("Description"):
                st.markdown("""
                **Method: Principal**
                
                **What it is**: Principal axis factoring (Principal) is a method of factor extraction that aims to explain the maximum amount of variance with each factor.
                
                **What it does**: It simplifies the factor analysis model by transforming the original variables into a smaller set of uncorrelated factors, which makes the underlying structure of the data more interpretable.
                
                **Rotation: Varimax**
                
                **What it is**: Varimax rotation is an orthogonal rotation method that simplifies the loadings of factors to make interpretation easier.
                
                **What it does**: It maximizes the variance of squared loadings of a factor across variables, which helps to achieve a clearer separation of factors, making it easier to identify which variables are most strongly associated with each factor.                
                """)

            n_factors = st.number_input("Enter the number of factors:", min_value=1, max_value=df2.shape[1], value=6)
            fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
            fa.fit(df2)
            fa_df = pd.DataFrame(fa.loadings_.round(2), index=df2.columns)
            
            sorted_loadings = []
            
            # Keep track of the rows already assigned
            assigned_rows = set()
            
            for i in range(n_factors):
                # Sort loadings for the current factor in descending order
                sorted_factor = fa_df.iloc[:, i].abs().sort_values(ascending=False)
            
                # Filter attributes with loadings above 0.5 and not already assigned
                high_loading_attrs = sorted_factor[sorted_factor > 0.4]
                high_loading_attrs = high_loading_attrs[~high_loading_attrs.index.isin(assigned_rows)]
            
                # Append these attributes to the list and mark them as assigned
                for attr in high_loading_attrs.index:
                    row = {'Attribute': attr}
                    for j in range(n_factors):
                        row[f'Factor {j+1}'] = fa_df.loc[attr, j]
                    sorted_loadings.append(row)
                    assigned_rows.add(attr)
            
            # Convert the sorted loadings list to a DataFrame
            sorted_loadings_df = pd.DataFrame(sorted_loadings)
            
            # Fill the rest of the attributes that were not included in any factor
            remaining_attrs = set(df2.columns) - assigned_rows
            remaining_rows = []
            for attr in remaining_attrs:
                row = {'Attribute': attr}
                for j in range(n_factors):
                    row[f'Factor {j+1}'] = fa_df.loc[attr, j]
                remaining_rows.append(row)
            
            # Concatenate the remaining rows to the sorted loadings DataFrame
            remaining_df = pd.DataFrame(remaining_rows)
            sorted_loadings_df = pd.concat([sorted_loadings_df, remaining_df], ignore_index=True)
            
            # Display the sorted loadings
            st.write("Factor Loadings:")
            st.write(sorted_loadings_df)
                    
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: Shows how much each variable contributes to each factor.
                        
                        **What it tells us**: High loadings indicate that a variable strongly influences the factor. It helps in understanding the underlying structure of the data.
                        """)

            # Download factor loadings as CSV
            csv = sorted_loadings_df.to_csv().encode('utf-8')
            st.download_button(label="Download Factor Loadings as CSV", data=csv, file_name='factor_loadings.csv', mime='text/csv')

            st.write("Factor Variance:")
            variance_df = pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var']).T
            st.write(variance_df)
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: The variance explained by each factor.
                        
                        **What it tells us**: Shows the proportion of total variance accounted for by each factor. Higher variance indicates a more significant factor.
                        """)

            # Communality
            st.write("Communality:")
            st.write(pd.DataFrame(fa.get_communalities(), index=df2.columns, columns=["Communality"]))
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: The proportion of variance in each variable explained by all the factors together.
                        
                        **What it tells us**: High communality values indicate that the variable is well represented by the factors extracted from the factor analysis.
                        """)

            # User-defined cluster names
            cluster_titles = st.text_input("Enter cluster names (comma-separated):", value="Efficacy,Supply and Samples,Patient Benefits,Cost and Coverage,Approval,MACE")
            cluster_titles = [x.strip() for x in cluster_titles.split(",")]
            factor_scores = fa.transform(df2)
            factor_scores = pd.DataFrame(factor_scores, columns=cluster_titles)
            st.write("Factor Scores:")
            st.write(factor_scores)
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: The scores (weights) assigned to each observation for each factor.
                        
                        **What it tells us**: Helps to interpret the relative importance of each factor for individual observations in the dataset.
                        """)

            # Split data
            X = factor_scores
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
                    
            # Models and Hyperparameters
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GBM': GradientBoostingClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42)
            }
            
            default_params = {
                'RandomForest': {'n_estimators': 500, 'max_depth': 5, 'max_features': 3},
                'GBM': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3},
                'XGBoost': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}
            }
            
            st.subheader("Model Training and Hyperparameter Tuning")
            
            # Model selection and parameters
            model_selection = st.selectbox("Select model:", models.keys())
            
            manual_params = {}
            if st.checkbox("Set hyperparameters manually"):
                if model_selection == 'RandomForest':
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=5)
                    manual_params['max_features'] = st.number_input("max_features", min_value=1, max_value=X.shape[1], value=3)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=100, max_value=1000, step=100, value=500)
                elif model_selection == 'GBM':
                    manual_params['learning_rate'] = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=3)
                elif model_selection == 'XGBoost':
                    manual_params['learning_rate'] = st.number_input("learning_rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
                    manual_params['n_estimators'] = st.number_input("n_estimators", min_value=50, max_value=500, step=50, value=100)
                    manual_params['max_depth'] = st.number_input("max_depth", min_value=1, max_value=20, value=3)
            
            # GridSearchCV
            grid_search_params = st.checkbox("Use GridSearchCV for hyperparameter tuning")
            if grid_search_params:
                st.write(f"Define GridSearchCV parameters for {model_selection}:")
                param_grid = {}
                if model_selection == 'RandomForest':
                    param_grid = {
                        'max_depth': st.multiselect("max_depth", [2, 3, 5, 10, 15], default=[3]),
                        'max_features': st.multiselect("max_features", list(range(1, X.shape[1] + 1)), default=[3]),
                        'n_estimators': st.multiselect("n_estimators", [100, 200, 500], default=[500])
                    }
                elif model_selection == 'GBM':
                    param_grid = {
                        'learning_rate': st.multiselect("learning_rate", [0.01, 0.1, 0.2], default=[0.1]),
                        'n_estimators': st.multiselect("n_estimators", [100, 200, 300], default=[100]),
                        'max_depth': st.multiselect("max_depth", [3, 5, 7], default=[3])
                    }
                elif model_selection == 'XGBoost':
                    param_grid = {
                        'learning_rate': st.multiselect("learning_rate", [0.01, 0.1, 0.2], default=[0.1]),
                        'n_estimators': st.multiselect("n_estimators", [100, 200, 300], default=[100]),
                        'max_depth': st.multiselect("max_depth", [3, 5, 7], default=[3])
                    }
            
                st.write(f"Running GridSearchCV for {model_selection}...")
                grid_search = GridSearchCV(estimator=models[model_selection], param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                st.write("Best Hyperparameters found by GridSearchCV:")
                st.write(best_params)
                final_params = best_params
            else:
                final_params = manual_params if manual_params else default_params[model_selection]
            
            st.write("Current Hyperparameters used:")
            st.write(final_params)
            
            model = models[model_selection].set_params(**final_params)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            cf_train = confusion_matrix(y_train, y_train_pred)
            cf_test = confusion_matrix(y_test, y_test_pred)
            TN_train, FP_train, FN_train, TP_train = cf_train.ravel()
            TN_test, FP_test, FN_test, TP_test = cf_test.ravel()
            
            st.write("""**Train Data Metrics:**""")
            st.write(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
            st.write(f"Sensitivity: {TP_train / (TP_train + FN_train)}")
            st.write(f"Specificity: {TN_train / (TN_train + FP_train)}")
            
            st.write("""**Test Data Metrics:**""")
            st.write(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
            st.write(f"Sensitivity: {TP_test / (TP_test + FN_test)}")
            st.write(f"Specificity: {TN_test / (TN_test + FP_test)}")
            
            st.write("""**Classification Report:**""")
            st.text(classification_report(y_test, y_test_pred))
            
            # Feature importance
            st.subheader("Feature Importance")

            # Feature Importance
            imp_df = pd.DataFrame({"varname": X_train.columns, "Importance": model.feature_importances_ * 100})
            imp_df.sort_values(by="Importance", ascending=False, inplace=True)
            #st.write("Feature Importance:")
            st.write(imp_df)

            # Plotting Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(imp_df)), imp_df["Importance"], align='center')
            plt.yticks(range(len(imp_df)), imp_df["varname"])
            plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            st.pyplot(plt)

            # Add explanation for feature importance
            with st.expander("Description"):
                        st.markdown("""
                        **What it is**: Feature importance is a measure of the influence each feature has on the predictions made by the model.
                        
                        **What it tells us**: Higher importance values indicate that the feature has a greater impact on the model's decision-making process.
                        
                        **How to interpret it**: Features with higher importance scores contribute more significantly to the prediction outcomes. This can help identify which variables are most influential in determining the target variable.
                        """)
            
            # Button to display ROC Curve
            if st.button("Show ROC Curve"):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                st.pyplot(plt)
            
            # Button to display Trees for RandomForest, GBM, and XGBoost
            if st.button("Show Trees"):
                if model_selection == 'RandomForest':
                    # Select one of the trees to display
                    st.write("Displaying a single tree from the RandomForest ensemble:")
                    estimator = model.estimators_[0]
                    dot_data = StringIO()
                    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                    special_characters=True, feature_names=X.columns, class_names=model.classes_.astype(str))
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    st.graphviz_chart(graph.to_string())
            
                elif model_selection == 'GBM':
                    # Select one of the trees to display
                    st.write("Displaying a single tree from the GBM ensemble:")
                    estimator = model.estimators_[0, 0]
                    dot_data = StringIO()
                    export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True,
                                    special_characters=True, feature_names=X.columns, class_names=model.classes_.astype(str))
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    st.graphviz_chart(graph.to_string())
            
                elif model_selection == 'XGBoost':
                    # Select one of the trees to display
                    st.write("Displaying a single tree from the XGBoost ensemble:")
                    booster = model.get_booster()
                    dot_data = xgb.to_graphviz(booster, num_trees=0)
                    st.graphviz_chart(dot_data.source)

if __name__ == "__main__":
    main()
