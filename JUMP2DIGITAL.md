<div style="background-color: #121212; padding: 10px;">
<img src="Images/header.png" alt="jum2digital" style="max-width:100%;">
</div>

<div style="background-color: #F5F5F5; padding: 20px; border: 1px solid #CCCCCC;">

### Project Description

This analysis centers on multiple datasets pertaining to Barcelona. Each dataset provides insights into distinct facets of the city—ranging from housing costs to environmental factors and public safety.

#### Datasets:

1. **Barcelona Rent**:
    - **Description**: Data about the average monthly rent (€/month) and by surface area (€/m2) in the city of Barcelona for the year 2017.
    - **Source**: [Open Data Barcelona](https://opendata-ajuntament.barcelona.cat/data/es/dataset/est-mercat-immobiliari-lloguer-mitja-mensual/resource/0a71a12d-55fa-4a76-b816-4ee55f84d327)
    - **File**: `2017_Alquiler_precio_trim.csv`

    
2. **Noise Exposure**:
    - **Description**: Information about the population's exposure to noise levels from the Strategic Noise Map of Barcelona.
    - **Source**: [Open Data Barcelona](https://opendata-ajuntament.barcelona.cat/data/es/dataset/poblacio-exposada-mapa-estrategic-soroll/resource/3846500e-72aa-4780-967f-f09aa184eaba)
    - **File**: `2017_Poblacio_exposada_barris_Mapa_Estrategic_Soroll_BCN_LONG.csv`

    
3. **Traffic Accidents**:
    - **Description**: Data about traffic accidents managed by the Urban Guard in Barcelona.
    - **Source**: [Open Data Barcelona](https://opendata-ajuntament.barcelona.cat/data/ca/dataset/accidents_causa_conductor_gu_bcn/resource/1a05cdd4-4844-41a5-872d-a0824d11b517?inner_span=True)
    - **File**: `2017_ACCIDENTS_CAUSA_CONDUCTOR_GU_BCN_.csv`

The objective of this analysis is to comprehensively evaluate these datasets to elucidate the socioeconomic and environmental dynamics shaping Barcelona.

</div>



<div style="background-color:#F5F5F5; border-left: 5px solid black; padding: 0.8em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Import requisites</h4>
</div>


```python
# Import required packages

# Data packages
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Visualization Packages
from matplotlib import pyplot as plt 
import seaborn as sns
%matplotlib inline

```

<div style="background-color:#F5F5F5; border-left: 5px solid black; padding: 0.8em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Auxiliary Functions</h4>
</div>


```python
def bold_print(text: str) -> str:
    """Returns the input string wrapped in ANSI escape codes that make it
    appear bold and dark when printed to a terminal for improving notebook
    visualization"""
    bold_text = "\033[1m" + "\033[90m" + text + "\033[0m"
    return bold_text
```

<div style="background-color:#F5F5F5; border-left: 5px solid black; padding: 0.8em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Load the Data</h4>      
</div>

<div style="background-color:#F5F5F5; padding-left: 1.5em; color: black;">
    - Let's first load the datasets   
</div>


```python
df_rent_prices = pd.read_csv("Data/2017_lloguer_preu_trim.csv")
print(bold_print(' Shape of df_rent_prices:'), df_rent_prices.shape)
print('\n',bold_print('First five entries of df_rent_prices:'))
df_rent_prices.head()
```

**Shape of df_rent_prices:** (584, 8)
    
**First five entries of df_rent_prices:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Any</th>
      <th>Trimestre</th>
      <th>Codi_Districte</th>
      <th>Nom_Districte</th>
      <th>Codi_Barri</th>
      <th>Nom_Barri</th>
      <th>Lloguer_mitja</th>
      <th>Preu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>Lloguer mitjà mensual (Euros/mes)</td>
      <td>734.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>2</td>
      <td>el Barri Gòtic</td>
      <td>Lloguer mitjà mensual (Euros/mes)</td>
      <td>905.26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>3</td>
      <td>la Barceloneta</td>
      <td>Lloguer mitjà mensual (Euros/mes)</td>
      <td>722.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>4</td>
      <td>Sant Pere, Santa Caterina i la Ribera</td>
      <td>Lloguer mitjà mensual (Euros/mes)</td>
      <td>895.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>2</td>
      <td>Eixample</td>
      <td>5</td>
      <td>el Fort Pienc</td>
      <td>Lloguer mitjà mensual (Euros/mes)</td>
      <td>871.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_noise = pd.read_csv("Data/2017_poblacio_exposada_barris_mapa_estrategic_soroll_bcn_long.csv")
print(bold_print(' Shape of df_noise:'), df_noise.shape)
print('\n',bold_print('First five entries of df_noise:'))
df_noise.head()
```

Shape of df_noise:[0m (18980, 7)
    
First five entries of df_noise:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Codi_Districte</th>
      <th>Nom_Districte</th>
      <th>Codi_Barri</th>
      <th>Nom_Barri</th>
      <th>Concepte</th>
      <th>Rang_soroll</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>TOTAL_D</td>
      <td>&lt;40 dB</td>
      <td>7.73%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>TOTAL_D</td>
      <td>40-45 dB</td>
      <td>26.98%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>TOTAL_D</td>
      <td>45-50 dB</td>
      <td>7.38%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>TOTAL_D</td>
      <td>50-55 dB</td>
      <td>11.97%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Ciutat Vella</td>
      <td>1</td>
      <td>el Raval</td>
      <td>TOTAL_D</td>
      <td>55-60 dB</td>
      <td>19.85%</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_accidents = pd.read_csv("Data/2017_accidents_causa_conductor_gu_bcn_.csv")
print(bold_print(' Shape of df_accidents:'), df_accidents.shape)
print('\n',bold_print('First five entries of df_accidents:'))
df_accidents.head()
```

    [1m[90m Shape of df_accidents:[0m (11091, 20)
    
     [1m[90mFirst five entries of df_accidents:[0m
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numero_expedient</th>
      <th>Codi_districte</th>
      <th>Nom_districte</th>
      <th>Codi_barri</th>
      <th>Nom_barri</th>
      <th>Codi_carrer</th>
      <th>Nom_carrer</th>
      <th>Num_postal</th>
      <th>Descripcio_dia_setmana</th>
      <th>NK_Any</th>
      <th>Mes_any</th>
      <th>Nom_mes</th>
      <th>Dia_mes</th>
      <th>Hora_dia</th>
      <th>Descripcio_torn</th>
      <th>Descripcio_causa_conductor</th>
      <th>Coordenada_UTM_X_ED50</th>
      <th>Coordenada_UTM_Y_ED50</th>
      <th>Longitud</th>
      <th>Latitud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017S004939</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Motors                                        ...</td>
      <td>43-51</td>
      <td>Dimarts</td>
      <td>2017</td>
      <td>6</td>
      <td>Juny</td>
      <td>6</td>
      <td>8</td>
      <td>Desobeir altres senyals</td>
      <td>Matí</td>
      <td>427585.89</td>
      <td>4577869.16</td>
      <td>2.191767</td>
      <td>41.411606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017S007937</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Joan XXIII / Martí i Franquès                 ...</td>
      <td>NaN</td>
      <td>Dimarts</td>
      <td>2017</td>
      <td>9</td>
      <td>Setembre</td>
      <td>26</td>
      <td>9</td>
      <td>Gir indegut o sense precaució</td>
      <td>Matí</td>
      <td>426505.49</td>
      <td>4581655.96</td>
      <td>2.199239</td>
      <td>41.419635</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017S004739</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Corts Catalanes                               ...</td>
      <td>900</td>
      <td>Dilluns</td>
      <td>2017</td>
      <td>5</td>
      <td>Maig</td>
      <td>29</td>
      <td>22</td>
      <td>Manca d'atenció a la conducció</td>
      <td>Nit</td>
      <td>432587.59</td>
      <td>4584475.05</td>
      <td>2.186875</td>
      <td>41.412198</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017S008779</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Número 6 Zona Franca / A Zona Franca          ...</td>
      <td>NaN</td>
      <td>Dilluns</td>
      <td>2017</td>
      <td>10</td>
      <td>Octubre</td>
      <td>23</td>
      <td>22</td>
      <td>Desobeir altres senyals</td>
      <td>Nit</td>
      <td>427519.99</td>
      <td>4575229.36</td>
      <td>2.190955</td>
      <td>41.406769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017S004350</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Desconegut</td>
      <td>-1</td>
      <td>Aguilar                                       ...</td>
      <td>7-9</td>
      <td>Dimarts</td>
      <td>2017</td>
      <td>5</td>
      <td>Maig</td>
      <td>16</td>
      <td>14</td>
      <td>Altres</td>
      <td>Tarda</td>
      <td>430758.19</td>
      <td>4586316.94</td>
      <td>2.186557</td>
      <td>41.409004</td>
    </tr>
  </tbody>
</table>
</div>



<div style="background-color:#F5F5F5; border-left: 5px solid black; padding: 0.8em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Exploratory Analysis</h4>      
</div>

<div style="background-color:#F5F5F5; padding-left: 1.5em; color: black;">
    - Let's check for missing values and review the datasets structure for each dataset  
</div>


```python
# Dataset info
df_rent_prices.info()
df_noise.info()
df_accidents.info()

# Check for missing values in each column
missing_values_1 = df_rent_prices.isnull().sum()
missing_values_2 = df_noise.isnull().sum()
missing_values_3 = df_accidents.isnull().sum()

print('\n',bold_print('Missing values df_rent_prices:'),'\n',missing_values_1,'\n')
print('\n',bold_print('Missing values df_noise:'),'\n', missing_values_2,'\n')
print('\n',bold_print('Missing values df_accidents:'),'\n', missing_values_3,'\n')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 584 entries, 0 to 583
    Data columns (total 8 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Any             584 non-null    int64  
     1   Trimestre       584 non-null    int64  
     2   Codi_Districte  584 non-null    int64  
     3   Nom_Districte   584 non-null    object 
     4   Codi_Barri      584 non-null    int64  
     5   Nom_Barri       584 non-null    object 
     6   Lloguer_mitja   584 non-null    object 
     7   Preu            546 non-null    float64
    dtypes: float64(1), int64(4), object(3)
    memory usage: 36.6+ KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18980 entries, 0 to 18979
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   Codi_Districte  18980 non-null  int64 
     1   Nom_Districte   18980 non-null  object
     2   Codi_Barri      18980 non-null  int64 
     3   Nom_Barri       18980 non-null  object
     4   Concepte        18980 non-null  object
     5   Rang_soroll     18980 non-null  object
     6   Valor           18980 non-null  object
    dtypes: int64(2), object(5)
    memory usage: 1.0+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11091 entries, 0 to 11090
    Data columns (total 20 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   Numero_expedient            11091 non-null  object 
     1   Codi_districte              11091 non-null  int64  
     2   Nom_districte               11091 non-null  object 
     3   Codi_barri                  11091 non-null  int64  
     4   Nom_barri                   11091 non-null  object 
     5   Codi_carrer                 11091 non-null  int64  
     6   Nom_carrer                  11091 non-null  object 
     7   Num_postal                  11089 non-null  object 
     8   Descripcio_dia_setmana      11091 non-null  object 
     9   NK_Any                      11091 non-null  int64  
     10  Mes_any                     11091 non-null  int64  
     11  Nom_mes                     11091 non-null  object 
     12  Dia_mes                     11091 non-null  int64  
     13  Hora_dia                    11091 non-null  int64  
     14  Descripcio_torn             11091 non-null  object 
     15  Descripcio_causa_conductor  11091 non-null  object 
     16  Coordenada_UTM_X_ED50       11091 non-null  float64
     17  Coordenada_UTM_Y_ED50       11091 non-null  float64
     18  Longitud                    11091 non-null  float64
     19  Latitud                     11091 non-null  float64
    dtypes: float64(4), int64(7), object(9)
    memory usage: 1.7+ MB
    
     [1m[90mMissing values df_rent_prices:[0m 
     Any                0
    Trimestre          0
    Codi_Districte     0
    Nom_Districte      0
    Codi_Barri         0
    Nom_Barri          0
    Lloguer_mitja      0
    Preu              38
    dtype: int64 
    
    
     [1m[90mMissing values df_noise:[0m 
     Codi_Districte    0
    Nom_Districte     0
    Codi_Barri        0
    Nom_Barri         0
    Concepte          0
    Rang_soroll       0
    Valor             0
    dtype: int64 
    
    
     [1m[90mMissing values df_accidents:[0m 
     Numero_expedient              0
    Codi_districte                0
    Nom_districte                 0
    Codi_barri                    0
    Nom_barri                     0
    Codi_carrer                   0
    Nom_carrer                    0
    Num_postal                    2
    Descripcio_dia_setmana        0
    NK_Any                        0
    Mes_any                       0
    Nom_mes                       0
    Dia_mes                       0
    Hora_dia                      0
    Descripcio_torn               0
    Descripcio_causa_conductor    0
    Coordenada_UTM_X_ED50         0
    Coordenada_UTM_Y_ED50         0
    Longitud                      0
    Latitud                       0
    dtype: int64 
    
    

<div style="background-color:#F5F5F5; padding: 0.8em; padding-left: 2em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Rent Prices Dataset (df_rent_prices):</h4>
    This dataset provides insights into the average rental prices in different districts and neighborhoods of a city, presumably Barcelona, for the year 2017.

**Columns and Data Types:**

- **Any** (int64): Year (e.g., 2017).
- **Trimestre** (int64): Quarter of the year, ranging from 1 to 4.
- **Codi_Districte** (int64): Code associated with each district.
- **Nom_Districte** (object): Name of the district.
- **Codi_Barri** (int64): Code for each neighborhood within the district.
- **Nom_Barri** (object): Name of the neighborhood.
- **Lloguer_mitja** (object): Type of rental average (likely indicating "Average monthly rent" in Euros, but there's a display issue).
- **Preu** (float64): Average monthly rental price in Euros.
    
**38 missing values in the Preu (Price) column**
</div>




<div style="background-color:#F5F5F5; padding: 0.8em; padding-left: 2em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Noise Levels Dataset (df_noise):</h4>
    This dataset provides information on the percentage of the population exposed to various noise levels in different districts and neighborhoods of Barcelona.

**Columns and Data Types:**

- **Codi_Districte** (int64): Code associated with each district.
- **Nom_Districte** (object): Name of the district.
- **Codi_Barri** (int64): Code for each neighborhood within the district.
- **Nom_Barri** (object): Name of the neighborhood.
- **Concepte** (object): Concept, which seems to indicate the total day.
- **Rang_soroll** (object): Noise level range in decibels (dB).
- **Valor** (object): Percentage of the population exposed to the corresponding noise level.

    
**0 missing values in the dataset**
</div>

<div style="background-color:#F5F5F5; padding: 0.8em; padding-left: 2em; color: black;">
    <h4 style="margin-top: 0.3em; margin-bottom: 0.3em;">Noise Levels Dataset (df_noise):</h4>
    This dataset provides information on the percentage of the population exposed to various noise levels in different districts and neighborhoods of Barcelona.

**Columns and Data Types:**

- **Codi_Districte** (int64): Code associated with each district.
- **Nom_Districte** (object): Name of the district.
- **Codi_Barri** (int64): Code for each neighborhood within the district.
- **Nom_Barri** (object): Name of the neighborhood.
- **Concepte** (object): Concept, which seems to indicate the total day.
- **Rang_soroll** (object): Noise level range in decibels (dB).
- **Valor** (object): Percentage of the population exposed to the corresponding noise level.

    
**0 missing values in the dataset**
</div>
