
import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
st.set_option('deprecation.showPyplotGlobalUse', False)


from sklearn.decomposition import PCA #Feature Reduction Method.
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
  st.title("Classify Interactively")
  
  def get_dataset(dataset_name):

    if dataset_name == 'Iris':
      data = datasets.load_iris()
      X = data.data
      y = data.target

    elif dataset_name == 'Breast Cancer':
      data = datasets.load_breast_cancer()
      X = data.data
      y = data.target

    elif dataset_name =="Wine":
      data = datasets.load_wine()
      X = data.data
      y = data.target

    else:
      st.write("Upload Your Dataset")
      uploaded_file = st.file_uploader("Choose a file")
      if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        target = st.selectbox("Select Target Variable : ", ['private'])

        container = st.beta_container()
        all = st.checkbox("Select all", key = '1')
 
        if all:
          predictors = container.multiselect("Select one or more Predictors:",list(df)[1:], list(df)[1:])
        else:
          predictors =  container.multiselect("Select one or more Predictors:", list(df)[1:],list(df)[1:4])


        
        st.write("Target Column unique values : ", sorted(df.private.unique()), "This will be Label Encoded : Set :  Yes : 1 and No : 0 ")
        
        df['target_code'] = df[target].apply(lambda x: 0 if x=='No' else 1)

        X = df[predictors]

        y = df['target_code']

        #Plotting of Custom Data Frame :
        # Sub header
        st.subheader("Basic Exploratory Data Analysis of selected Columns of the Dataset : ")

        st.write("Select the columns for Pairwise Plots : ")

        container1 = st.beta_container()
        all = st.checkbox("Select all (It'll make you wait a lot) ", key='2')
        if all:
          col = container1.multiselect("Select one or more Columns:",list(df), list(df))
        else:
          col =  container1.multiselect("Select one or more Columns:", list(df),list(df)[0:2])

        def EDA():
          if st.checkbox("Check Box to view Seaborn Pairplot (You'll have to wait a bit)",value=False, key = '3'):
            fig = plt.figure()
            fig = sns.pairplot(df[col],hue = target)
            return(st.pyplot(fig))
          else:
            return("Check Box to see Plots")
            
        EDA()

    return X,y

  dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Upload Custom Dataset"))
  X,y = get_dataset(dataset_name)
  
  st.subheader("Basic Information about the Dataset : ")
  st.write("Dataset Chosen : ", dataset_name)
  
  st.write("Shape of Dataset : ", X.shape)
  st.write("Number of Classes to Classify : ", len(np.unique(y)))

  classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "DT", "RF", "NB"))

  if classifier_name == "KNN":
    st.write("Classifier Chosen : K-Nearest Neighbors : ", classifier_name)

  elif classifier_name == "SVM":
    st.write("Classifier Chosen : Support Vector Machines : ", classifier_name)

  elif classifier_name == "DT":
    st.write("Classifier Chosen : Decision Trees : ", classifier_name)

  elif classifier_name == "RF":
    st.write("Classifier Chosen : Random Forest Ensemble : ", classifier_name)

  else:
    st.write("Classifier Chosen : Gaussian Naive Bayes : ", classifier_name)
  


  def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
      K = st.sidebar.slider("K", 1, 15)
      params["K"] = K
    
    elif classifier_name == "SVM":
      C = st.sidebar.slider("C", 0.01, 10.0) 
      params["C"] = C
      kernel = st.sidebar.selectbox("kernel", ('rbf', 'linear', 'poly'))
      params["kernel"] = kernel

    elif classifier_name == "RF":
      max_depth = st.sidebar.slider("max_depth", 2, 15)
      n_estimators = st.sidebar.slider("n_estimators", 1, 100)
      params['max_depth'] = max_depth
      params['n_estimators'] = n_estimators

    elif classifier_name == "DT":
      splitter = st.sidebar.selectbox("splitter", ('best', 'random'))
      criterion = st.sidebar.selectbox("criterion", ('gini', 'entropy'))
      max_depth = list(range(1,51))
      max_depth.insert(0, None)
      max_depth = st.sidebar.selectbox("max_depth", max_depth)
      params['max_depth'] = max_depth
      params['criterion'] = criterion
      params['splitter'] = splitter

    else:
      params['var_smoothing'] = 1e-9

    return params

  params = add_parameter_ui(classifier_name)

  def get_classifier(clf_name, params):

    if classifier_name == "KNN":
      clf = KNeighborsClassifier(n_neighbors = params["K"])
    
    elif classifier_name == "SVM":
      clf = SVC(C = params["C"], kernel = params["kernel"])

    elif classifier_name == "RF":
      clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                   max_depth = params["max_depth"], random_state = 1234)

    elif classifier_name == "DT":
      clf = DecisionTreeClassifier(criterion = params["criterion"], splitter = params["splitter"], max_depth = params["max_depth"])

    else:
      clf = GaussianNB(var_smoothing = params['var_smoothing'])

    return clf

  clf = get_classifier(classifier_name, params)

  # Do Classification
  test_size = st.sidebar.slider("Test Size : ", 0.1,0.9)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state = 1234, stratify = y)

  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)*100
  precision = precision_score(y_test, y_pred, average = 'weighted')*100
  recall = recall_score(y_test, y_pred, average = 'weighted')*100
  F1 = (2*precision *recall)/(precision+recall)

  clf_rep = classification_report(y_test, y_pred,output_dict = True)
  clf_rep_df = pd.DataFrame(clf_rep).transpose()


  st.write(" ")
  st.subheader("Results of the Classification : ")

  st.write(f"Classifer used : {clf}.")
  #st.write(" ")
  st.write(f"Accuracy Score : {accuracy:.2f}%")
  #st.write(" ")
  st.write(f"Precision Score : {precision:.2f}%")
  #st.write(" ")
  st.write(f"Recall Score : {recall:.2f}%")
  #st.write(" ")
  st.write(f"F1 - Score : {F1:.2f}%")
  #st.write(" ")
  st.write("Classification Report")
  st.write(clf_rep_df)
  st.write(" ")

  def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
      st.subheader('Confusion Matrix')
      plot_confusion_matrix(clf, X_test, y_test)
      st.pyplot()

    if('ROC Curve' in metrics_list and len(np.unique(y)) == 2):
      st.subheader('ROC Curve')
      plot_roc_curve(clf, X_test, y_test)
      st.pyplot()

    if('ROC Curve' in metrics_list and len(np.unique(y)) > 2):
      st.write("Notification : No ROC  Curve for unique classes more than 2.")

    if('Precision-Recall Curve' in metrics_list and len(np.unique(y)) == 2):
      st.subheader('Notification : Precision-Recall Curve')
      plot_precision_recall_curve(clf, X_test, y_test)
      st.pyplot()
    
    if('Precision-Recall Curve' in metrics_list and len(np.unique(y)) > 2):
      st.write("Notification : No Precision-Recall Curve for unique classes more than 2.")

  metrics = st.sidebar.multiselect('Select your Metrics to plot : ', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key ='metrics')
  
  #Plot Metrics
  st.subheader("Plot Metrics")
  plot_metrics(metrics)

  #Plot
  st.subheader("Visualization")
  pca = PCA(3)
  X_projected = pca.fit_transform(X)

  x1 = X_projected[:,0]
  x2 = X_projected[:,1]
  x3 = X_projected[:,2]

  st.subheader("Plotly Interactive 2D Plot : 2 Principal Components")
  
  # Add traces
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x1, y=x2,
                    mode='markers', marker=dict(size=10, color=y, colorscale='Viridis', opacity=0.8)
                    ))
  fig.layout.update(title_text = dataset_name + ' Dataset Visualization', width=800, height=800, xaxis_title = 'Principal Component - 1' ,yaxis_title='Principal Component - 2')
  st.plotly_chart(fig)


  st.subheader("Plotly Interactive 3D Plot : 3 Principal Components")
  
  fig = go.Figure(data=[go.Scatter3d(x=x1, y=x2, z=x3, mode='markers',marker=dict(size=5, color=y, colorscale='Viridis', opacity=0.8) )])

  # tight layout
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), )
  st.plotly_chart(fig)

if __name__ == '__main__':
  main()