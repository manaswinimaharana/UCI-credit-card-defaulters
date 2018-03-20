# UCI-credit-card-defaulters

UCI credit card defaulters demo<br>
Created by: Mansi Maharana(mansi@cloudera.com)

<b>Status:</b> In Progress <br>
<b>Use Case:</b> Build a prediction model for banking credit card data defaulter based on the UCI credit card defaulters data from a Taiwan bank. Covers end-to-end lifecycle from Statistical derivation from data, choosing a algorithm, building a model, using a ML pipeline, deriving model summary, cross validation & visualization  

<b>Steps</b>:<br>
1. Open a CDSW terminal and run setup.sh<br>
2. Create a Python 2 Session and run input-statistics.py<br>
3. Return to the Python 2 Session and run build-model-sparkML.py<br>
4. Return to the Python 2 Session and run build-model-pandas.py<br>
4. When finished, run cleanup.sh in your terminal<br>

<b>Recommended Session Sizes</b>: 1 CPU, 2 GB RAM

<b>Recommended Jobs/Pipeline</b>:<br>
input-statistics.py --> build-model.py 

<b>Notes</b>: <br>
1. input-statistics.py --> Methods to derive statistical information from data<br>
2. build-model-sparkML.py --> Use Transformations like OneHotEncoding & Logistics regression to build predictive model <br>
3. build-model-pandas.py --> Look how the same model can be build using Pandas and scaled out using Spark <br>

<b>Estimated Runtime</b>: <br>
input-statistics.py--> approx TBD min <br>
build-model-sparkML.py --> < TBD min <br>
build-model-pandas.py --> approx TBD min <br>

<b>Demo Script</b><br>


<b>Related Content</b>:<br>
https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
