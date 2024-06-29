import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO

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
    st.title("Non-Linear Classification Analysis Model_Commnts")

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
            st.write(f"KMO Test Statistic: {kmo_model:.3f}")
            st.markdown("""
            **What it is**: A measure of how suited data is for factor analysis. It assesses the proportion of variance among variables that might be common variance.
            
            **What it tells us**: A KMO value closer to 1 indicates that a factor analysis may be useful. Values below 0.6 generally indicate the data is not suitable for factor analysis.
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
            st.markdown("""
            **What it is**: A graph showing the eigenvalues of the factors in descending order.
            
            **What it tells us**: Helps to determine the number of factors to retain by identifying the point where the curve starts to flatten (the "elbow").
            """)

            # Heatmap of correlation matrix
            plt.figure(figsize=(20, 10))
            sns.heatmap(df2.corr(), cmap="Reds", annot=True)
            st.pyplot(plt)
            st.markdown("""
            **What it is**: A visual representation of the correlation matrix where the strength of correlation is represented by color intensity.
            
            **What it tells us**: Helps to identify the strength and direction of relationships between variables. High correlation values indicate multicollinearity.
            """)

            # Variance Inflation Factor (VIF)
            df2_with_const = add_constant(df2)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = df2_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(df2_with_const.values, i) for i in range(df2_with_const.shape[1])]
            st.write("Variance Inflation Factor (VIF):")
            st.write(vif_data)
            st.markdown("""
            **What it is**: Measures the increase in variance of the estimated regression coefficients due to collinearity.
            
            **What it tells us**: VIF values above 10 indicate high multicollinearity, suggesting that the predictor variables are highly correlated and may not be suitable for regression analysis.
            """)

            # Factor Analysis
            st.subheader("Factor Analysis")

            if st.checkbox("Click to select method and rotation"):
                rotation_options = ["None", "Varimax", "Promax", "Quartimax", "Oblimin"]
                rotation = st.selectbox("Select rotation:", rotation_options)
                method_options = ["Principal", "Minres", "ML", "GLS", "OLS"]
                method = st.selectbox("Select method:", method_options)
                if rotation == "None":
                    rotation = None
                if method == "Principal":
                    method = "principal"
            else:
                rotation = "varimax"
                method = "principal"

            st.write(f"Method: {method}, Rotation: {rotation}")

            n_factors = st.number_input("Enter the number of factors:", min_value=1, max_value=df2.shape[1], value=6)
            fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
            fa.fit(df2)
            fa_df = pd.DataFrame(fa.loadings_.round(2), index=df2.columns)
            st.write("Factor Loadings:")
            st.write(fa_df)
            st.markdown("""
            **What it is**: Shows how much each variable contributes to each factor.
            
            **What it tells us**: High loadings indicate that a variable strongly influences the factor. It helps in understanding the underlying structure of the data.
            """)

            # Download factor loadings as CSV
            csv = fa_df.to_csv().encode('utf-8')
            csv_b64 = base64.b64encode(csv).decode()  # Convert to base64
            
            button_style = """
                <style>
                    .download-button {
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
                <a download="factor_loadings.csv" href="data:text/csv;base64,{csv_b64}" class="download-button">Download Factor Loadings as CSV</a>
            """.format(csv_b64=csv_b64)
            
            components.html(button_style, height=60)

            # Display variance explained by each factor
            variance_df = pd.DataFrame(fa.get_factor_variance().round(2), index=["Variance", "Proportional Var", "Cumulative Var"])
            st.write("Variance Explained by Each Factor:")
            st.write(variance_df)
            st.markdown("""
            **What it is**: Displays the variance explained by each factor.
            
            **What it tells us**: Higher variance indicates that the factor explains a larger proportion of the data's total variance.
            """)

            # Model fit and validity tests
            st.header("Model Fit and Validity Tests")

            # Adjusted R-squared
            X = add_constant(df2)
            model = sm.OLS(y, X).fit()
            st.write("**Adjusted R-squared:**")
            st.write(model.rsquared_adj)
            st.markdown("""
            **What it is**: A version of the R-squared statistic adjusted for the number of predictors in the model.
            
            **What it tells us**: Indicates the proportion of variance explained by the model, adjusted for the number of predictors. Higher values indicate better fit.
            """)

            # Split data into training and test sets
            st.header("Random Forest Classification")
            test_size = st.slider("Test size (as a proportion):", min_value=0.1, max_value=0.9, value=0.2)
            random_state = st.number_input("Random state (for reproducibility):", value=42)

            X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=test_size, random_state=random_state)

            # Display the training and test set shapes
            st.write("Training set shape:")
            st.write(X_train.shape)
            st.write("Test set shape:")
            st.write(X_test.shape)

            # Hyperparameter tuning
            st.subheader("Hyperparameter Tuning")
            tune = st.checkbox("Perform hyperparameter tuning")
            if tune:
                st.write("Select hyperparameter ranges for tuning:")
                n_estimators_range = st.slider("Number of estimators:", min_value=10, max_value=300, value=(10, 100))
                max_depth_range = st.slider("Max depth:", min_value=1, max_value=50, value=(1, 10))
                param_grid = {
                    'n_estimators': np.arange(n_estimators_range[0], n_estimators_range[1] + 1, 10),
                    'max_depth': np.arange(max_depth_range[0], max_depth_range[1] + 1, 1)
                }
                rf = RandomForestClassifier(random_state=random_state)
                grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                st.write("Best parameters found:", best_params)
                model = RandomForestClassifier(**best_params, random_state=random_state)
            else:
                model = RandomForestClassifier(random_state=random_state)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", accuracy)

            # Show classification report
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Show confusion matrix
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)

            # ROC curve
            st.write("ROC Curve:")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            # Feature importance
            st.subheader("Feature Importance")

            # Calculate feature importance
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)

            # Create a DataFrame for feature importance
            feature_importance_df = pd.DataFrame({
                'Feature': df2.columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            # Display feature importance table and plot side by side
            col1, col2 = st.columns(2)

            with col1:
                st.write("Feature Importance Table:")
                st.dataframe(feature_importance_df)

            with col2:
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
                plt.yticks(range(len(sorted_idx)), np.array(df2.columns)[sorted_idx])
                plt.xlabel('Importance')
                plt.title('Feature Importance')
                st.pyplot(plt)

            # Add explanation for feature importance
            st.markdown("""
            **What it is**: Feature importance is a measure of the influence each feature has on the predictions made by the model.
            
            **What it tells us**: Higher importance values indicate that the feature has a greater impact on the model's decision-making process.
            
            **How to interpret it**: Features with higher importance scores contribute more significantly to the prediction outcomes. This can help identify which variables are most influential in determining the target variable.
            """)
            
            # Display tree structure
            if st.checkbox("Show Tree Structure"):
                estimator = model.estimators_[0]
                dot_data = StringIO()
                export_graphviz(estimator, out_file=dot_data, feature_names=df2.columns, class_names=[str(i) for i in model.classes_], rounded=True, proportion=False, precision=2, filled=True)
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                st.graphviz_chart(graph.to_string())

if __name__ == "__main__":
    main()
