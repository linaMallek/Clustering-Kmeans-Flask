# Clustering with Flask

This project is a web application that allows users to upload CSV files and perform clustering using hierarchical clustering and k-means algorithms. The application returns the clustering results, including the silhouette score and the number of clusters.

## Features
- **File Upload**: Upload CSV files for clustering.
- **Hierarchical Clustering**: Automatically determines the number of clusters and computes the silhouette score.
- **K-Means Clustering**: Allows users to specify the number of clusters (k) and computes the silhouette score.

## Requirements
- Python 3.x
- Flask
- Pandas
- Matplotlib
- Scikit-learn

## How to Run
1. Install the required dependencies:
   ```bash
   pip install flask pandas matplotlib scikit-learn
