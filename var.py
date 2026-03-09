"""

VAR plot for timeframes


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.api import VAR

# Ha szükséges, importáld be az adataidat, például:
# from your_data import gdp_log_for_coint_diff  # Cseréld le a megfelelő adatfájlt

# Define the timeframes for the analysis
timeframes = {
    "1995-2004": ("1995-01-01", "2004-12-31"),
    "2004-2014": ("2004-01-01", "2014-12-31"),
    "2014-2023": ("2014-01-01", "2023-12-31"),
}

# Helper function to filter data for each timeframe
def filter_by_timeframe(df, start_date, end_date):
    return df[(df.index >= start_date) & (df.index <= end_date)]

# Helper function for computing the FEVD
def compute_fevd(df, optimal_lag=10):
    model = VAR(df)
    results = model.fit(optimal_lag)
    fevd_result = results.fevd(10)
    fevd_matrix = fevd_result.decomp
    return np.round(fevd_matrix, 4)

# Helper function to calculate From, To, and Net connectedness
def calculate_connectedness(fevd_matrix_rounded, columns):
    from_connectedness = np.sum(fevd_matrix_rounded, axis=1) - np.diag(fevd_matrix_rounded)
    to_connectedness = np.sum(fevd_matrix_rounded, axis=0) - np.diag(fevd_matrix_rounded)
    net_connectedness = from_connectedness - to_connectedness

    connectedness_df = pd.DataFrame({
        'From Connectedness': from_connectedness,
        'To Connectedness': to_connectedness,
        'Net Connectedness': net_connectedness
    }, index=columns)
    
    return connectedness_df

# Function to create visualizations
def create_plots(fevd_matrix_rounded, connectedness_df, label, columns):
    # Interactive heatmap plot
    fevd_df = pd.DataFrame(fevd_matrix_rounded, index=columns, columns=columns)
    fig = px.imshow(fevd_df,
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    labels=dict(x="To", y="From", color="Spillover"),
                    title=f"Generalized FEVD Matrix ({label})")
    fig.update_layout(width=800, height=600)
    fig.show()

    # Static heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(fevd_matrix_rounded.T, cmap="viridis", interpolation="none")  # Transpose to match From and To
    plt.colorbar(label="Spillover")
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)
    plt.title(f"Spillover ({label})")
    plt.tight_layout()
    plt.show()

    # Bar plot for From, To and Net Connectedness
    plt.figure(figsize=(10, 6))
    connectedness_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f"From, To, and Net Connectedness ({label})")
    plt.xlabel("Countries")
    plt.ylabel("Connectedness")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function to perform spillover analysis over different timeframes
def perform_spillover_analysis(gdp_log_for_coint_diff):
    connectedness_results = {}

    for label, (start_date, end_date) in timeframes.items():
        # Filter data for the current timeframe
        filtered_data = filter_by_timeframe(gdp_log_for_coint_diff, start_date, end_date)

        # Compute FEVD matrix for the current timeframe
        fevd_matrix_rounded = compute_fevd(filtered_data)

        # Calculate From, To, and Net connectedness
        connectedness_df = calculate_connectedness(fevd_matrix_rounded, gdp_log_for_coint_diff.columns)

        # Store the connectedness results
        connectedness_results[label] = connectedness_df

        # Print the connectedness results for the current timeframe
        print(f"\n--- {label} Connectedness ---")
        print(connectedness_df)

        # Create plots for the current timeframe
        create_plots(fevd_matrix_rounded, connectedness_df, label, gdp_log_for_coint_diff.columns)

# Example usage: 
# from
