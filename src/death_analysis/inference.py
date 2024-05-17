"""
Importing the necessary libraries.
"""

# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Reading the values and storing it in a dataframe using pandas
death_df = pd.read_csv(
    "https://raw.githubusercontent.com/ArunMathew7/DAV-5400/main/Analysis%20of%20death%20by%20selected%20causes%20from%202014%20to%202019/Monthly_Counts_of_Deaths_by_Select_Causes__2014-2019.csv",
    encoding="unicode_escape",
)

# Define features and target variable of model
X = death_df.drop(["Jurisdiction of Occurrence", "Year", "Month", "All Cause"], axis=1)
y = death_df["All Cause"]


class InferenceAnalysis:
    """
    This class contains the functions to analyse the plots regarding research questions
    and get the conclusions of all the research questions.
    """

    def __init__(self):
        """ """
        pass

    def research_question2():
        """
        Code visualizes seasonal variations in mortality for different diseases or causes over a span of months.
        By plotting each disease or cause separately, the viewer can observe how each one's mortality rate changes throughout the year.
        Patterns and trends specific to certain diseases or causes can be identified. For example, some diseases may exhibit seasonal spikes or declines.
        """

        # Select the columns representing diseases and the month
        disease_columns = death_df.columns[4:]

        # Group the dataset by month and calculate the mean number of deaths for each disease
        monthly_disease_deaths = death_df.groupby("Month")[disease_columns].mean()

        # Plotting using matplotlib
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        # Plot each disease separately
        for column in disease_columns:
            plt.plot(
                monthly_disease_deaths.index,
                monthly_disease_deaths[column],
                label=column,
            )

        # Add plot title and labels
        plt.title(
            "Seasonal Variations in Mortality for Different Diseases or Causes using Matplotlib"
        )
        plt.xlabel("Month")
        plt.ylabel("Mean Number of Deaths")
        plt.legend(title="Disease or Cause", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)

        # Plotting using Seaborn
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 2)
        sns.lineplot(data=monthly_disease_deaths, dashes=False, markers=True)
        plt.title(
            "Seasonal Variations in Mortality for Different Diseases or Causes using Seaborn"
        )
        plt.xlabel("Month")
        plt.ylabel("Mean Number of Deaths")
        plt.legend(title="Disease or Cause", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.show()

    def research_question1():
        """
        Code visualizes the trends in monthly deaths due to influenza and pnuemonia over the years.
        Analysing this graph it is seen that average deaths of influenza and pnuemoia is a dip in the period of fall months
        and its high in winter period, which shows that cold weather has a role for these diseases.
        """

        # Select relevant columns
        df_subset = death_df[["Year", "Month", "Influenza and Pneumonia"]]

        # Group data by year and month and calculate the mean for Influenza and Pneumonia
        monthly_mean = df_subset.groupby(["Year", "Month"]).mean().reset_index()

        # Plotting using matplotlib
        # Plotting
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        sns.lineplot(
            data=monthly_mean,
            x="Month",
            y="Influenza and Pneumonia",
            hue="Year",
            marker="o",
        )

        plt.title(
            "Monthly Trends in Deaths due to Influenza and Pneumonia Over the Years using Seaborn"
        )
        plt.xlabel("Month")
        plt.ylabel("Average Deaths")
        plt.legend(title="Year", loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.tight_layout()

        # Plotting using Seaborn
        plt.subplot(1, 2, 2)
        # Loop through unique years and plot the lines
        for year in monthly_mean["Year"].unique():
            year_data = monthly_mean[monthly_mean["Year"] == year]
            plt.plot(
                year_data["Month"],
                year_data["Influenza and Pneumonia"],
                marker="o",
                label=str(year),
            )

        plt.title(
            "Monthly Trends in Deaths due to Influenza and Pneumonia Over the Years using Matplotlib"
        )
        plt.xlabel("Month")
        plt.ylabel("Average Deaths")
        plt.legend(title="Year", loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.tight_layout()
        plt.show()

    def prediction_model():
        """
        Trains a prediction model and visualizes actual vs predicted values.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardize the features
                (
                    "model",
                    RandomForestRegressor(),
                ),  # Use a RandomForestRegressor a|s the model
            ]
        )

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        print(f"Cross-Validation Scores: {cv_scores}")

        pipeline.fit(X_train, y_train)

        test_score = pipeline.score(X_test, y_test)
        print(f"Model R^2 Score on Test Set: {test_score}")

        return pipeline

    def model_prediction_graph():
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardize the features
                (
                    "model",
                    RandomForestRegressor(),
                ),  # Use a RandomForestRegressor a|s the model
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        # Create a DataFrame with actual and predicted values
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})

        # Plotting using matplotlib
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x="Actual", y="Predicted", data=results_df, color="blue", alpha=0.7
        )

        # Add a diagonal line for reference
        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            linestyle="--",
            color="red",
            linewidth=2,
        )

        # Set plot labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values using Matplotlib")

        plt.subplot(1, 2, 2)
        # plotting using seaborn
        sns.scatterplot(
            x="Actual", y="Predicted", data=results_df, color="blue", alpha=0.7
        )

        # Add a diagonal line for reference
        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            linestyle="--",
            color="orange",
            linewidth=2,
        )

        # Set plot labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values using Seaborn")

        plt.show()

    def feature_importance():
        """
        Visualizes feature importance scores for the prediction model.
        """
        pipeline = InferenceAnalysis.prediction_model()
        # 'pipeline' is your trained model
        feature_importance = pipeline.named_steps["model"].feature_importances_

        # Create a DataFrame to display feature importance
        feature_importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importance}
        )

        # Sort the DataFrame by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        # Display the top contributors to all-cause mortality
        print(feature_importance_df)

    def rq1_model():
        """
        Compares actual vs predicted seasonal mortality rates.
        """
        # 'pipeline' is your trained model
        pipeline = InferenceAnalysis.prediction_model()
        # Make predictions on the entire dataset
        predicted_mortality = pipeline.predict(X)

        # Create a DataFrame to compare predicted vs actual mortality
        comparison_df = pd.DataFrame(
            {
                "Actual_Mortality": y,
                "Predicted_Mortality": predicted_mortality,
                "Month": death_df["Month"],
                "Year": death_df["Year"],
            }
        )
        # Group by month and calculate the average actual and predicted mortality
        monthly_avg_comparison = comparison_df.groupby("Month").mean()

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 2, 1)
        # ploting using matplotlib
        plt.plot(
            monthly_avg_comparison.index,
            monthly_avg_comparison["Actual_Mortality"],
            label="Actual Mortality",
        )
        plt.plot(
            monthly_avg_comparison.index,
            monthly_avg_comparison["Predicted_Mortality"],
            label="Predicted Mortality",
        )
        plt.xlabel("Month")
        plt.ylabel("Mortality Rate")
        plt.title("Actual vs Predicted Monthly Mortality Rates using Matplotlib")
        plt.legend()

        plt.subplot(1, 2, 2)
        # Set the plotting style using Seaborn
        sns.set(style="whitegrid")

        # Create a Seaborn line plot
        sns.lineplot(
            x="Month",
            y="Actual_Mortality",
            data=monthly_avg_comparison,
            label="Actual Mortality",
        )
        sns.lineplot(
            x="Month",
            y="Predicted_Mortality",
            data=monthly_avg_comparison,
            label="Predicted Mortality",
        )
        plt.xlabel("Month")
        plt.ylabel("Mortality Rate")
        plt.title("Actual vs Predicted Monthly Mortality Rates using Seaborn")
        plt.legend()
        plt.show()

    def rq2_model():
        """
        Visualizes feature importance scores for the prediction model using both Matplotlib and Seaborn.
        """
        #'pipeline' is your trained model
        pipeline = InferenceAnalysis.prediction_model()

        feature_importance = pipeline.named_steps["model"].feature_importances_
        features = X.columns

        plt.figure(figsize=(10, 6))
        # plotting using matplotlib
        plt.barh(features, feature_importance)
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance Scores using Matplotlib")

        # Create a horizontal bar plot using Seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=features, palette="viridis")

        # Set plot labels and title
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance Scores using Seaborn")
        plt.show()

        feature_importance_dict = dict(zip(features, feature_importance))
        sorted_importance = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        print("Feature Importance:")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance}")
