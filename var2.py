"""

Imports

"""
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
print(gdp_real)
gdp_log = np.log(gdp_real) # logging the GDP data
gdp_log_for_coint = gdp_log.dropna() # NaN values are dropped



"""

ADF and Phillips-Perron unit root tests are performed on the log GDP data
If the p-value is less than 0.05, we reject the null hypothesis of a unit root, indicating stationarity.
The ADF test statistic should be less than the critical value for stationarity.
The PP test statistic should be less than the critical value for stationarity.


"""

print("\n--- ADF and Phillips-Perron Unit Root Tests ---\n")
adf_results = {}
pp_results = {}

# First for the non-differenced data
for col in gdp_log_for_coint.columns:
    print(f"--- {col} ---")
    # ADF
    adf_test = adfuller(gdp_log_for_coint[col])
    print(f"ADF Test Statistic: {adf_test[0]:.4f}, p-value: {adf_test[1]:.4f}")
    adf_results[col] = adf_test[1]
    
    # PP
    pp_test = PhillipsPerron(gdp_log_for_coint[col])
    print(f"PP Test Statistic: {pp_test.stat:.4f}, p-value: {pp_test.pvalue:.4f}")
    pp_results[col] = pp_test.pvalue
    print()

# For the differenced data
gdp_log_for_coint_diff=gdp_log_for_coint.diff().dropna()
for col in gdp_log_for_coint_diff.columns:
    print(f"--- {col} ---")
    # ADF
    adf_test = adfuller(gdp_log_for_coint_diff[col])
    print(f"ADF Test Statistic: {adf_test[0]:.4f}, p-value: {adf_test[1]:.4f}")
    adf_results[col] = adf_test[1]
    
    # PP
    pp_test = PhillipsPerron(gdp_log_for_coint_diff[col])
    print(f"PP Test Statistic: {pp_test.stat:.4f}, p-value: {pp_test.pvalue:.4f}")
    pp_results[col] = pp_test.pvalue
    print()

print("\n--- Descriptive Statistics for log real GDP ---\n")

summary_stats = []

for col in gdp_log_for_coint.columns:
    
    series = gdp_log_for_coint[col].dropna()
    mean = series.mean()
    mode = series.mode()
    median = series.median()
    maximum = series.max()
    minimum = series.min()
    std_dev = series.std()
    skewness = skew(series)
    kurt = kurtosis(series, fisher=False)  # standard kurtosis, not excess
    jb_stat, jb_p = jarque_bera(series)
    observations = series.count()
    print(gdp_log_for_coint.index)
    print(gdp_log_for_coint.shape)
    summary_stats.append([col, mean, median, maximum, minimum, std_dev,
                          skewness, kurt, jb_stat, jb_p, observations])

# Adatok DataFrame-be rendezése és kerekítés
columns = ["Country", "Mean", "Median", "Max", "Min", "Std. Dev.",
           "Skewness", "Kurtosis", "Jarque-Bera", "JB p-value", "Obs"]

stats_df = pd.DataFrame(summary_stats, columns=columns)
stats_df.set_index("Country", inplace=True)
stats_df = stats_df.round(4)

# Kiírás
print(stats_df)

"""

VAR model fitting and lag selection

Johansen-test (Cointegration test), MaxEigen and Trace statistics
If the p-value is less than 0.05, we reject the null hypothesis of a unit root, indicating stationarity.
If the test statistic is greater than the critical value, we reject the null hypothesis of no cointegration.
If the full rank is returned, it indicates that the variables are not cointegrated.

"""

model = VAR(gdp_log_for_coint_diff)
lag_selection = model.select_order(maxlags=12)  # Maximum lag order to consider
print(lag_selection.summary())


print("\nLag according to AIC:", lag_selection.aic)
print("Lag according to BIC:", lag_selection.bic)
print("Lag according to HQIC:", lag_selection.hqic)
print("Lag according to FPE:", lag_selection.fpe)
'''
if lag_selection.aic==lag_selection.bic==lag_selection.hqic==lag_selection.fpe:
    optimal_lag = lag_selection.aic
else:
    print("Optimal lag is not the same for all criteria. Please check the results.")
'''
optimal_lag = lag_selection.aic
print(optimal_lag )
# Johansen-test (Cointegration test), MaxEigen and Trace statistics
print("\n--- Johansen Cointegration Test ---\n")
johansen_result = coint_johansen(gdp_log_for_coint_diff, det_order=-1, k_ar_diff=12) # k_ar_diff is the number of lags in the VAR model, det_order is the deterministic trend order (-1 for no deterministic trend)

print("Trace Statistics:\n", johansen_result.lr1)
print("Critical Values (90%, 95%, 99%):\n", johansen_result.cvt) # 95% critical values are significant in terms of cointegration
trace_rank = sum(johansen_result.lr1 > johansen_result.cvt[:, 1]) 
print(f"Cointegration Rank: {trace_rank}")

print("\nMaxEigen Statistics:\n", johansen_result.lr2)
print(johansen_result.eig)
last_eigenvector = johansen_result.evec[:, -1]  # vagy model.evec[:, -1]



print("Critical Values (90%, 95%, 99%):\n", johansen_result.cvm)
max_eigen_rank = sum(johansen_result.lr2 > johansen_result.cvm[:, 1]) # 95% critical values are significant in terms of cointegration
print(f"Cointegration Rank based on MaxEigen: {max_eigen_rank}")


results=model.fit(optimal_lag) # fitting the model with the selected lag order

print("\n--- VAR Model Summary ---\n")
#print(results.summary()) 


"""


Checking the stability of the VAR model
The roots of the characteristic polynomial should be outside the unit circle for stability, so the invers e of the roots should be less than 1.



"""

roots = results.roots
inverse_roots = 1 / roots

print("Inverse roots (abs):")
print(np.abs(inverse_roots))
stability_check = np.abs(inverse_roots)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(np.real(inverse_roots), np.imag(inverse_roots), color='blue', label='Roots')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', label='One')
plt.gca().add_artist(circle)
plt.title("VAR Modell Stability Check")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.grid(True)
plt.show()

# Stability check
if np.all(stability_check > 1):
    print("The VAR model is stable.")
else:
    print("The VAR model is not stable.")


"""


Autocorrelation test (Ljung-Box, Jarque-Bera and Shapiro-Wilk tests) for residuals
The null hypothesis is that the residuals are not autocorrelated.


"""
resid = results.resid
lbvalue, pvalue = acorr_ljungbox(resid.values[:, 0], lags=[12], return_df=True)

print(lbvalue)
for i, col in enumerate(resid.columns):
    print(f"\n--- Ljung-Box Test for residual {col} ---")
    lb_test = acorr_ljungbox(resid.iloc[:, i], lags=[12], return_df=True)
    print(lb_test)



for i, col in enumerate(resid.columns):
    
    # Jarque-Bera test
    jb_stat, jb_p_value = jarque_bera(resid[col])
    print(f"Jarque-Bera Test Statistic for {col}: {jb_stat}")
    print(f"Jarque-Bera Test p-value for {col}: {jb_p_value}")

    # Shapiro-Wilk test
    stat, p_value = shapiro(resid[col])
    print(f"Shapiro-Wilk Test Statistic for {col}: {stat}")
    print(f"Shapiro-Wilk Test p-value for {col}: {p_value}")



# Max correlation in residuals
correlation_matrix_residuals = resid.corr()
print(correlation_matrix_residuals)
max_correlation_residuals = correlation_matrix_residuals.abs().max().max() 
print(f"Max correlation in residuals: {max_correlation_residuals}")


"""

FEVD (Forecast Error Variance Decomposition)

H=20->longer horizon, H=10->shorter horizon
Total connectedness index is calculated as the sum of the off-diagonal elements of the FEVD matrix divided by the number of variables (the value of total ccnnectedness is increasing with the H).
Tables for from-to connectedness and net connectedness.

"""


def generalized_fevd(results, H=1):

    
    N = results.neqs
    print(f"Number of variables: {N}")
    p = results.k_ar
    print(f"Number of lags: {p}")
    sigma = results.sigma_u.values  
    #print(f"Sigma shape: {sigma.shape}",sigma)
    coefs = results.coefs
    #print(f"Coefs shape: {coefs.shape}",coefs)

    A = [np.eye(N)] # Unity matrix for A_0
    
    
    for h in range(1, H):
        A_h = np.zeros((N, N))
        for i in range(1, min(p, h)+1):
            #print(p,h)
            A_h += coefs[i-1] @ A[h-i]
            #print(A_h)
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

    return fevd / fevd.sum(axis=1, keepdims=True)

# FEVD for total connectedness
fevd_matrix = generalized_fevd(results, H=20)
total_connectedness = (np.sum(fevd_matrix) - np.trace(fevd_matrix)) / fevd_matrix.shape[0] * 100
print(f"Total connectedness index: {total_connectedness:.2f}%")

# FEVD for from-to connectedness
from_connectedness = np.sum(fevd_matrix, axis=1) - np.diag(fevd_matrix)  # Sum of rows, excluding diagonal
to_connectedness = np.sum(fevd_matrix, axis=0) - np.diag(fevd_matrix)  # Sum of columns, excluding diagonal

# Table for from-to connectedness
fevd_matrix_rounded = np.round(fevd_matrix, 4)

from_to_df = pd.DataFrame(fevd_matrix_rounded, index=gdp_log_for_coint_diff.columns, columns=gdp_log_for_coint_diff.columns)
from_connectedness = np.sum(fevd_matrix_rounded, axis=1) - np.diag(fevd_matrix_rounded)  
to_connectedness = np.sum(fevd_matrix_rounded, axis=0) - np.diag(fevd_matrix_rounded)  

from_to_df["From Connectedness"] = from_connectedness # From connectedness
from_to_df["To Connectedness"] = to_connectedness # To connectedness
net_connectedness = to_connectedness - from_connectedness # Net connectedness
from_to_df["Net Connectedness"] = net_connectedness

# Results
print("\n--- From-To, Net Spillover ---\n")
print(from_to_df)

"""

Plot

"""


# Heatmap for spillover matrix
plt.figure(figsize=(8, 6))
plt.imshow(fevd_matrix, cmap="viridis", interpolation="none")  
plt.colorbar(label="Spillover")
plt.xticks(range(len(gdp_log_for_coint_diff.columns)), gdp_log_for_coint_diff.columns, rotation=45)
plt.yticks(range(len(gdp_log_for_coint_diff.columns)), gdp_log_for_coint_diff.columns)
plt.title("Spillover (H = 20)")
plt.tight_layout()
plt.show()

fevd_df = pd.DataFrame(fevd_matrix, index=gdp_log_for_coint_diff.columns, columns=gdp_log_for_coint_diff.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(fevd_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Spillover'})  
plt.title("Spillover Heatmap", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Log GDP time series for each country
plt.figure(figsize=(12, 6))
for country in gdp_log_for_coint_diff.columns:
    plt.plot(gdp_log_for_coint_diff.index, gdp_log_for_coint_diff[country], label=country)

plt.title("Log GDP Time Series")
plt.xlabel("Time")
plt.ylabel("log(GDP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Connectedness bar diagram
connectedness_df = pd.DataFrame({
    'To Connectedness': np.round(to_connectedness, 4),  
    'From Connectedness': np.round(from_connectedness, 4),  
    'Net Connectedness': np.round(net_connectedness, 4)}, index=gdp_log_for_coint_diff.columns)

# Print the results
print("\nFrom, To, and Net Connectedness:")
print(connectedness_df)


plt.figure(figsize=(12, 7))
ax = connectedness_df.plot(kind='bar', width=0.8, color=['skyblue', 'orange', 'lightgreen'], edgecolor='black', ax=plt.gca())
ax.set_title("From, To and Net Connectedness", fontsize=16, fontweight='bold')
ax.set_xlabel("Countries", fontsize=14)
ax.set_ylabel("Connectedness", fontsize=14)
ax.set_xticklabels(connectedness_df.index, rotation=45, ha='right', fontsize=12)
ax.legend(title="Connectedness Types", fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
plt.tight_layout()
plt.show()

# Ábra beállításai

print(np.log(gdp_real))
plt.figure(figsize=(14, 8))

# Országonkénti log GDP idősor kirajzolása
for country in gdp_real.columns:
    plt.plot(
        (gdp_real.index), 
        np.log(gdp_real[country]), 
        label=country
    )
    min_val = np.log(gdp_real[country].min())
    max_val = np.log(gdp_real[country].max())
    print(f"{country}: log(GDP) min = {min_val:.2f}, max = {max_val:.2f}")

# Címek és feliratok
plt.title("Logaritmizált GDP idősorok (nem differenciált)", fontsize=16)
plt.xlabel("Év", fontsize=13)
plt.ylabel("log(GDP)", fontsize=13)
plt.legend(title="Országok")
plt.grid(True)
plt.tight_layout()
plt.show()
###########################################################################

# Stílus beállítása
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# Színek generálása az országok számához igazítva
colors = sns.color_palette("tab10", n_colors=len(gdp_log_for_coint.columns))

# Ábra készítése
plt.figure(figsize=(16, 9))

# Vonalgrafikon minden országhoz
for i, country in enumerate(gdp_log_for_coint.columns):
    plt.plot(
        gdp_log_for_coint.index,
        gdp_log_for_coint[country],
        label=country,
        color=colors[i],
        linewidth=2.2,
        marker='o',
        markersize=4,
        alpha=0.9
    )

# Cím és tengelyek
plt.title("Logaritmizált reál GDP idősorok (1995–)", fontsize=20, weight='bold')
plt.xlabel("Év", fontsize=16)
plt.ylabel("log(GDP)", fontsize=16)

# Y tengely dinamizálása – nagyítjuk a tartományt kicsit a kontraszt érdekében
ymin = gdp_log_for_coint.min().min() - 0.5
ymax = gdp_log_for_coint.max().max() + 0.5
plt.ylim(ymin, ymax)

# Tengelyfeliratok és jelmagyarázat
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Ország", fontsize=11, title_fontsize=13, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Az adatok beolvasása CSV formátumból (az általad adott stringből)
data_str = """
country name,country code,indicator name,indicator code,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023
China,CHN,GDP (constant 2015 US$),NY.GDP.MKTP.KD,1831415106374.65,2013138309684.44,2199087464269.01,2371626806476.09,2553332587278.43,2770112908903.6,3001022138034.24,3275124420038.43,3603882407613.85,3968365429214.8,4420544471398.72,4982879973774.79,5691986693310.73,6241302053216.75,6827904909100.22,7554112071638.2,8275592638197.0,8926363432826.3,9619598215288.48,10333926843443.8,11061572618578.7,11819153423353.8,12640253743744.4,13493442283300.8,14296369668065.2,14616413683019.5,15851276922873.1,16318995784492.7,17175670911003.5
Czechia,CZE,GDP (constant 2015 US$),NY.GDP.MKTP.KD,118281953148.026,123212636048.58,122478487849.992,121996963422.938,123678521510.239,128638858397.276,132391536317.902,134395228578.688,138831395782.306,145406922302.962,154676655561.766,164921158026.322,173973393363.484,178518044738.046,169952170252.553,174565802530.616,177659773362.468,176290606026.729,176216894679.608,180173337872.859,189107698561.919,193988322699.324,204024435225.367,209798950840.998,217279912678.49,205753475034.301,214043320209.36,220137498815.997,219949654880.837
Germany,DEU,GDP (constant 2015 US$),NY.GDP.MKTP.KD,2612945784475.61,2640072232962.63,2689040748598.27,2745407383018.29,2803887763760.38,2884562510841.22,2931769559373.65,2925076028983.88,2909575203020.53,2943395181897.32,2969464751675.73,3083959477633.35,3173089212221.55,3201977117826.16,3024422218182.65,3149837977159.83,3268207907666.64,3283708733629.98,3296391223767.13,3367906396055.51,3423568450957.03,3502129441730.16,3597248136065.74,3637409365003.91,3673343090202.78,3522914640155.8,3652205602572.65,3702230996916.9,3692366831510.82
Hungary,HUN,GDP (constant 2015 US$),NY.GDP.MKTP.KD,79280778283.9448,79346116187.6974,81840061575.3627,85031827660.5837,87643177933.1084,91568886542.0243,95299483590.3448,99817905184.6813,103884735932.983,109083772922.95,113768132959.824,118257499574.879,118585513679.933,119776044677.938,111873380290.514,113077422068.682,115188393760.125,113748278485.318,115798616344.321,120699456740.623,125174166987.372,127929252772.234,133394359768.376,140547430034.099,147383974295.866,140770787502.349,150710923361.943,157618619387.289,156187937314.849
Poland,POL,GDP (constant 2015 US$),NY.GDP.MKTP.KD,218906804615.997,232246717071.23,247015068311.447,258431144714.153,270575301764.16,283174187558.032,286667833930.23,292117946677.278,302410346199.761,317805438387.054,328168510412.522,348521872927.444,372083512402.176,388394800371.402,398551823849.317,411181642127.603,432791139408.41,439333996551.267,442340098156.254,459682079851.439,480054118583.367,494605411858.337,520089746650.75,552573816155.989,577884228677.544,566120997479.772,605337233054.782,637150470884.733,638035044435.602
Slovak Republic,SVK,GDP (constant 2015 US$),NY.GDP.MKTP.KD,41585644886.9254,44130511304.3812,46551440693.9027,48332548394.3004,48103808997.7433,48483456571.093,49901674780.841,52105958464.3133,54636587612.3566,57581225947.9722,61315329304.8089,66788143655.4668,74013706557.0379,77983340638.8351,73690093822.7164,78694080172.9019,80710556609.3781,81977043383.7687,82553607304.9464,84789118228.7254,89178548717.3324,90915585456.2015,93529165116.4867,97328432384.4156,99543529380.5199,96969818956.4464,102523269375.456,102984289733.717,104403760583.519
United States,USA,GDP (constant 2015 US$),NY.GDP.MKTP.KD,11106673921000.0,11525703481000.0,12038266261000.0,12577957790000.0,13180243872000.0,13717679619000.0,13848757308000.0,14084248131000.0,14477988212000.0,15035068144000.0,15558822251000.0,15992063824000.0,16312522122000.0,16331051067000.0,15910281498000.0,16339094225000.0,16594704135000.0,16974575729000.0,17334068403000.0,17771549056000.0,18295019000000.0,18627887993000.0,19085691122935.2,19651869117582.5,20159639089698.1,19723580221938.2,20917853444668.3,21443388432051.0,22062578283266.8
"""

from io import StringIO

df = pd.read_csv(StringIO(data_str))

# Évek oszlopok
years = [str(y) for y in range(1995, 2024)]
print("\n--- Descriptive Statistics for log real GDP ---\n")

summary_stats = []

for idx, row in df.iterrows():
    country = row['country name']
    gdp_values = row[years].astype(float).dropna()
    log_gdp = np.log(gdp_values)
    mean = log_gdp.mean()
    median = log_gdp.median()
    maximum = log_gdp.max()
    minimum = log_gdp.min()
    std_dev = log_gdp.std()
    skewness = skew(log_gdp)
    kurt_val = kurtosis(log_gdp, fisher=False)  # standard kurtosis, not excess
    jb_stat, jb_p = jarque_bera(log_gdp)
    observations = log_gdp.count()
    summary_stats.append([country, mean, median, maximum, minimum, std_dev,
                          skewness, kurt_val, jb_stat, jb_p, observations])

# Adatok DataFrame-be rendezése és kerekítés
columns = ["Country", "Mean", "Median", "Max", "Min", "Std. Dev.",
           "Skewness", "Kurtosis", "Jarque-Bera", "JB p-value", "Obs"]

stats_df = pd.DataFrame(summary_stats, columns=columns)
stats_df.set_index("Country", inplace=True)
stats_df = stats_df.round(4)

# Kiírás
print(stats_df)
# DataFrame létrehozása


# Plot elkészítése
plt.figure(figsize=(12, 7))
for idx, row in df.iterrows():
    gdp_values = row[years].astype(float)
    plt.plot(years, np.log(gdp_values), label=row['country name'])

plt.xlabel('Year')
plt.ylabel('Log Real GDP (constant 2015 US$)')
plt.title('Logarithm of Real GDP (constant 2015 US$) from 1995 to 2023')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
fevd=results.fevd(10) # 20 is the horizon
print(fevd.summary())