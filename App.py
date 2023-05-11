# Import des bibliothèques nécessaires
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from fonction import *
from flask import send_file
from flask import Flask, render_template, request

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   
      if request.method == 'POST':
        f = request.files['file']
        k =int( request.form['k'])
        filename = f.filename    
        f.save(f.filename)
   

      # Chargement des données
      df = pd.read_csv(filename,header=None)
      X = df.values
      silhouette_avg, nbr_clusters=hierarchical_clustering(X)
      centroids, labels, silhouette_avgU=kmeans(X,nbr_clusters)
      centroids, labels, silhouette_avgT=kmeansT(X,k)


      return render_template('result.html', silhouette_avgU=silhouette_avgU, silhouette_avgT=silhouette_avgT, nbr_clusters=nbr_clusters, k=k)


if __name__ == '__main__':
    app.run(debug=True)
