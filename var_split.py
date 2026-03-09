import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import shapiro, jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import PhillipsPerron

"""
Input data and CSV file loading and selection
"""

df = pd.read_csv("WorldBank.csv", sep=",", encoding="utf-8-sig", header=2, on_bad_lines='skip')
df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip().str.lower().str.replace('"', '', regex=False)

countries_of_interest = ['USA', 'CHN', 'DEU', 'HUN', 'POL', 'CZE', 'SVK']
countries = ["China", "Czechia", "Germany", "Hungary", "Poland", "Slovak Republic", "United States"]
target_countries = ['Czechia', 'Hungary', 'Poland', 'Slovak Republic']
influential_countries = ['United States', 'China', 'Germany']


df.columns = df.columns.str.strip().str.lower()

if 'country code' in df.columns:

    df_filtered = df[df['country code'].isin(countries_of_interest)] # filtering
    df_filtered = df_filtered.loc[:, (df_filtered.columns[:4]).tolist() + [col for col in df_filtered.columns if col.isdigit() and 1995 <= int(col) <= 2023]]
    df_filtered.to_csv("filtered_data.csv", index=False) # save the new CSV file
    print("Filtered data saved to 'filtered_data.csv'.")

else:

    print("Error: 'Country Code' column not found in the dataset.") # check if the column exists


"""
The filtered date is loaded and the GDP data is transformed to quarterly frequency
"""

df = pd.read_csv('filtered_data.csv')
df_melted = df.melt(id_vars=['country name', 'country code', 'indicator name', 'indicator code'],var_name='Year', value_name='GDP')
df_melted['Year'] = df_melted['Year'].astype(int)
df_melted['GDP'] = pd.to_numeric(df_melted['GDP'], errors='coerce')
df_melted['date'] = pd.to_datetime(df_melted['Year'].astype(str) + '-12-31')


"""
Group by country and resample to quarterly frequency
"""

dfs = []
for country, group in df_melted.groupby('country name'):
    group = group.set_index('date')
    group = group[['GDP']]
    group = group.resample('QE').asfreq()  # Quarter End
    group['GDP'] = group['GDP'].interpolate(method='linear')
    group['country name'] = country
    dfs.append(group)

df_quarterly = pd.concat(dfs).reset_index()


gdp_real = df_quarterly.pivot(index='date', columns='country name', values='GDP') # real GDP data
gdp_log = np.log(gdp_real) # logging the GDP data
gdp_log_for_coint = gdp_log.dropna() # NaN values are dropped


# --- Function to encapsulate the VAR analysis ---
def run_var_analysis(data_log, title_suffix=""):
    """
    Performs VAR analysis including unit root tests, descriptive statistics,
    lag selection, cointegration test, stability check, residual diagnostics,
    FEVD, and plotting.

    Args:
        data_log (pd.DataFrame): Logged and pre-processed GDP data.
        title_suffix (str): Suffix for plot titles and print statements
                            to distinguish between periods.
    """
    print(f"\n--- Analysis for Period: {title_suffix} ---")

    # Ensure data is differenced for VAR modeling, as per your original code
    data_log_diff = data_log.diff().dropna()
    print(f"Data shape after differencing: {data_log_diff.shape}")

    if data_log_diff.empty:
        print(f"Skipping analysis for {title_suffix}: Differenced data is empty.")
        return

    # --- ADF and Phillips-Perron Unit Root Tests ---
    print(f"\n--- ADF and Phillips-Perron Unit Root Tests ({title_suffix}) ---\n")
    print("For non-differenced data:")
    for col in data_log.columns:
        if not data_log[col].empty:
            print(f"--- {col} ---")
            # ADF
            adf_test = adfuller(data_log[col])
            print(f"ADF Test Statistic: {adf_test[0]:.4f}, p-value: {adf_test[1]:.4f}")

            # PP
            pp_test = PhillipsPerron(data_log[col])
            print(f"PP Test Statistic: {pp_test.stat:.4f}, p-value: {pp_test.pvalue:.4f}")
            print()
        else:
            print(f"--- {col} --- No data for unit root test.")


    print("\nFor differenced data:")
    for col in data_log_diff.columns:
        if not data_log_diff[col].empty:
            print(f"--- {col} ---")
            # ADF
            adf_test = adfuller(data_log_diff[col])
            print(f"ADF Test Statistic: {adf_test[0]:.4f}, p-value: {adf_test[1]:.4f}")

            # PP
            pp_test = PhillipsPerron(data_log_diff[col])
            print(f"PP Test Statistic: {pp_test.stat:.4f}, p-value: {pp_test.pvalue:.4f}")
            print()
        else:
            print(f"--- {col} --- No data for differenced unit root test.")


    # --- Descriptive Statistics for log real GDP ---
    print(f"\n--- Descriptive Statistics for log real GDP ({title_suffix}) ---\n")
    summary_stats = []
    for col in data_log.columns:
        series = data_log[col].dropna()
        if not series.empty:
            mean = series.mean()
            median = series.median()
            maximum = series.max()
            minimum = series.min()
            std_dev = series.std()
            skewness = skew(series)
            kurt = kurtosis(series, fisher=False)
            jb_stat, jb_p = jarque_bera(series)
            observations = series.count()
            summary_stats.append([col, mean, median, maximum, minimum, std_dev,
                                  skewness, kurt, jb_stat, jb_p, observations])
        else:
            summary_stats.append([col] + [np.nan]*10) # Append NaNs if no data

    columns = ["Country", "Mean", "Median", "Max", "Min", "Std. Dev.",
               "Skewness", "Kurtosis", "Jarque-Bera", "JB p-value", "Obs"]
    stats_df = pd.DataFrame(summary_stats, columns=columns)
    stats_df.set_index("Country", inplace=True)
    stats_df = stats_df.round(4)
    print(stats_df)

    # --- VAR model fitting and lag selection ---
    model = VAR(data_log_diff)
    lag_selection = model.select_order()
    print(f"\n--- Lag Selection ({title_suffix}) ---\n")
    print(lag_selection.summary())

    optimal_lag = lag_selection.aic
    print(f"\nOptimal lag according to AIC ({title_suffix}): {optimal_lag}")

    # Johansen-test (Cointegration test)
    print(f"\n--- Johansen Cointegration Test ({title_suffix}) ---\n")
    # Adjust k_ar_diff for Johansen test based on the optimal_lag
    # k_ar_diff is the number of lags of the differenced series in the VECM,
    # which is optimal_lag - 1 if optimal_lag > 0, otherwise 0.
    johansen_k_ar_diff = max(1, optimal_lag) # Ensure at least 1 for Johansen test
    
    try:
        johansen_result = coint_johansen(data_log, det_order=-1, k_ar_diff=johansen_k_ar_diff)

        print("Trace Statistics:\n", johansen_result.lr1)
        print("Critical Values (90%, 95%, 99%):\n", johansen_result.cvt)
        trace_rank = sum(johansen_result.lr1 > johansen_result.cvt[:, 1])
        print(f"Cointegration Rank (Trace) ({title_suffix}): {trace_rank}")

        print("\nMaxEigen Statistics:\n", johansen_result.lr2)
        print("Critical Values (90%, 95%, 99%):\n", johansen_result.cvm)
        max_eigen_rank = sum(johansen_result.lr2 > johansen_result.cvm[:, 1])
        print(f"Cointegration Rank (MaxEigen) ({title_suffix}): {max_eigen_rank}")
    except ValueError as e:
        print(f"Could not perform Johansen Cointegration Test for {title_suffix}: {e}")
        print("This might happen if the time series is too short or all variables are already stationary.")

    # Fit the VAR model
    try:
        results = model.fit(optimal_lag)
        print(f"\n--- VAR Model Summary ({title_suffix}) ---\n")
        # print(results.summary()) # Uncomment if you want to see the full summary
    except ValueError as e:
        print(f"Could not fit VAR model for {title_suffix}: {e}")
        print("This often happens if there's not enough data for the chosen lag order after differencing.")
        return

    # --- Checking the stability of the VAR model ---
    roots = results.roots
    inverse_roots = 1 / roots

    print(f"\nInverse roots (abs) ({title_suffix}):")
    print(np.abs(inverse_roots))
    stability_check = np.abs(inverse_roots)

    plt.figure(figsize=(8, 6))
    plt.scatter(np.real(inverse_roots), np.imag(inverse_roots), color='blue', label='Roots')
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', label='Unit Circle')
    plt.gca().add_artist(circle)
    plt.title(f"VAR Model Stability Check ({title_suffix})")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.grid(True)
    plt.show()

    if np.all(stability_check < 1): # Roots should be inside the unit circle, so inverse roots outside
        print(f"The VAR model for {title_suffix} is stable (all inverse roots outside the unit circle).")
    else:
        print(f"The VAR model for {title_suffix} is not stable (at least one inverse root is inside or on the unit circle).")


    # --- Autocorrelation test (Ljung-Box, Jarque-Bera and Shapiro-Wilk tests) for residuals ---
    resid = results.resid
    print(f"\n--- Residual Diagnostics ({title_suffix}) ---")

    for i, col in enumerate(resid.columns):
        if not resid.iloc[:, i].empty:
            print(f"\n--- Ljung-Box Test for residual {col} ({title_suffix}) ---")
            lb_test = acorr_ljungbox(resid.iloc[:, i], lags=[12], return_df=True)
            print(lb_test)

            # Jarque-Bera test
            jb_stat, jb_p_value = jarque_bera(resid[col])
            print(f"Jarque-Bera Test Statistic for {col}: {jb_stat:.4f}")
            print(f"Jarque-Bera Test p-value for {col}: {jb_p_value:.4f}")

            # Shapiro-Wilk test
            try:
                stat, p_value = shapiro(resid[col])
                print(f"Shapiro-Wilk Test Statistic for {col}: {stat:.4f}")
                print(f"Shapiro-Wilk Test p-value for {col}: {p_value:.4f}")
            except Exception as e:
                print(f"Shapiro-Wilk Test could not be performed for {col}: {e}")
        else:
            print(f"\n--- No residuals to test for {col} ({title_suffix}) ---")

    correlation_matrix_residuals = resid.corr()
    print(f"\nCorrelation Matrix of Residuals ({title_suffix}):")
    print(correlation_matrix_residuals)
    max_correlation_residuals = correlation_matrix_residuals.abs().max().max()
    print(f"Max absolute correlation in residuals ({title_suffix}): {max_correlation_residuals:.4f}")


    # --- FEVD (Forecast Error Variance Decomposition) ---
    def generalized_fevd(results_var, H=1):
        N = results_var.neqs
        p = results_var.k_ar
        sigma = results_var.sigma_u.values
        coefs = results_var.coefs

        A = [np.eye(N)] # Unity matrix for A_0
        for h in range(1, H):
            A_h = np.zeros((N, N))
            for i in range(1, min(p, h)+1):
                A_h += coefs[i-1] @ A[h-i]
            A.append(A_h)

        fevd = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                num = 0
                denom = 0
                for h in range(H):
                    e_i = np.zeros(N); e_i[i] = 1
                    e_j = np.zeros(N); e_j[j] = 1
                    num += (e_i @ A[h] @ sigma @ e_j.T)**2 / sigma[j, j]
                    denom += e_i @ A[h] @ sigma @ A[h].T @ e_i.T
                if denom > 0:
                    fevd[i, j] = num / denom
                else:
                    fevd[i, j] = 0

        # Normalize rows to sum to 1
        row_sums = fevd.sum(axis=1, keepdims=True)
        # Avoid division by zero for rows that sum to zero (unlikely but good practice)
        fevd_normalized = np.where(row_sums == 0, 0, fevd / row_sums)
        return fevd_normalized

    fevd_matrix = generalized_fevd(results, H=1)
    total_connectedness = (np.sum(fevd_matrix) - np.trace(fevd_matrix)) / fevd_matrix.shape[0] * 100
    print(f"\nTotal connectedness index ({title_suffix}): {total_connectedness:.2f}%")

    from_connectedness = np.sum(fevd_matrix, axis=1) - np.diag(fevd_matrix)
    to_connectedness = np.sum(fevd_matrix, axis=0) - np.diag(fevd_matrix)

    fevd_matrix_rounded = np.round(fevd_matrix, 4)
    from_to_df = pd.DataFrame(fevd_matrix_rounded, index=data_log_diff.columns, columns=data_log_diff.columns)
    from_to_df["From Connectedness"] = from_connectedness
    from_to_df["To Connectedness"] = to_connectedness
    net_connectedness = to_connectedness - from_connectedness
    from_to_df["Net Connectedness"] = net_connectedness

    print(f"\n--- From-To, Net Spillover ({title_suffix}) ---\n")
    print(from_to_df)

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(fevd_matrix.T, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Spillover'})
    plt.xticks(range(len(data_log_diff.columns)), data_log_diff.columns, rotation=45, ha='right')
    plt.yticks(range(len(data_log_diff.columns)), data_log_diff.columns, rotation=0)
    plt.title(f"Spillover Heatmap (H = 20) ({title_suffix})", fontsize=14)
    plt.tight_layout()
    plt.show()

    connectedness_df = pd.DataFrame({
        'To Connectedness': np.round(to_connectedness, 4),
        'From Connectedness': np.round(from_connectedness, 4),
        'Net Connectedness': np.round(net_connectedness, 4)}, index=data_log_diff.columns)

    print(f"\nFrom, To, and Net Connectedness ({title_suffix}):")
    print(connectedness_df)

    plt.figure(figsize=(12, 7))
    ax = connectedness_df.plot(kind='bar', width=0.8, color=['skyblue', 'orange', 'lightgreen'], edgecolor='black', ax=plt.gca())
    ax.set_title(f"From, To and Net Connectedness ({title_suffix})", fontsize=16, fontweight='bold')
    ax.set_xlabel("Countries", fontsize=14)
    ax.set_ylabel("Connectedness", fontsize=14)
    ax.set_xticklabels(connectedness_df.index, rotation=45, ha='right', fontsize=12)
    ax.legend(title="Connectedness Types", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Log GDP time series for each country
    plt.figure(figsize=(12, 6))
    for country in data_log.columns:
        plt.plot(data_log.index, data_log[country], label=country)

    plt.title(f"Log GDP Time Series ({title_suffix})")
    plt.xlabel("Time")
    plt.ylabel("log(GDP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sns.heatmap(fevd_matrix.T, xticklabels=data_log.columns, yticklabels=data_log.columns, annot=True, cmap="YlGnBu")
    plt.title(f"Generalized FEVD Matrix ({title_suffix}) - Horizon = 10")
    plt.show()


# --- Data Splitting ---
gdp_log_pre_crisis = gdp_log_for_coint['1995-01-01':'2009-12-31']
gdp_log_post_crisis = gdp_log_for_coint['2010-01-01':'2023-12-31']

# --- Run analysis for each period ---
#run_var_analysis(gdp_log_pre_crisis, "1995-2007 (Pre-Crisis)")
run_var_analysis(gdp_log_post_crisis, "2008-2023 (Post-Crisis)")