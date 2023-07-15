# import libraries
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import data
def wrangle(filepath):

    """Read SCF data file into ``DataFrame``.

    Returns only households with small businesses that earn  less than $500,000.

    Parameters
    ----------
    filepath : str
        Location of CSV file.
    """
    
    # import data
    df = pd.read_csv(filepath)
    # create mask for small business owners
    mask = (df['HBUS']==1) & (df["INCOME"]<500_000)
    # subset data for small business owners
    df = df[mask]
    
    return df

# import wrangled data
df = wrangle("/Users/Blurryface/Documents/GitHub/Data_Science_Portfolio/6_Customer_Segmentation/Data/scfp2019excel.zip")


# initialize dash app
app = Dash(__name__)
server = app.server

# create application layout
# Set application layout
app.layout = html.Div(
    [
#         application title
        html.H1(children="Survey of Consumer Finances", style={'textAlign':'center'}),
#         bar chart element
        html.H2("High Variance Features"),
#         bar chart
        dcc.Graph(id="bar-chart"),
#         paragraph to explain trimmed and not trimmed
        html.P("Trimmed variance removes the bottom and top 10% of observations (outliers)."),
#         radio buttons
        dcc.RadioItems(
            options = [
                {"label":"trimmed", "value":True},
                {"label":"not trimmed", "value":False}
            ],
            value = True,
            id = "trim-button"
        ),
        
#         KMeans Clustering
        html.H2("K-Means Clustering"),
        
#         pca scatterplot
        dcc.Graph(id="pca-scatter"),
        
#         k slider
        html.H3("Number of Clusters(k)"),
        dcc.Slider(min = 2, max = 12, step = 1, value = 2, id = "k-slider"),
        html.Div(id = "metrics")
        

    ]
)

# Variance bar chart: backend layer
def get_high_var_features(trimmed = True, return_feat_names=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    
#     calculate variance
    if trimmed:
        top_five_features = (
            df.apply(trimmed_var).sort_values().tail(5)
        )
    else:
         top_five_features = (
            df.var().sort_values().tail(5)
        )
        
#     Extract names
    if return_feat_names:
        top_five_features = top_five_features.index.to_list()
    
    
    return top_five_features

# variance bar chart: connection layer
@callback(
    Output("bar-chart", "figure"), Input("trim-button", "value")
)
def serve_bar_chart(trimmed = True): #variance barchat: frontend

    """Returns a horizontal bar chart of five highest-variance features.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.
    """
#     get features
    top_five_features = get_high_var_features(trimmed = trimmed, return_feat_names = False)
    
#     build bar chart
    fig = px.bar(
        x = top_five_features,
        y = top_five_features.index,
        orientation = 'h'
    )
    fig.update_layout(
        xaxis_title = "Variance",
        yaxis_title = "Feature"
    )
    
    return fig


# K-means model and slider: backend
def get_model_metrics(trimmed = True, k = 2, return_metrics = False):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
#     get high var features
    features = get_high_var_features(trimmed = trimmed, return_feat_names=True)
#     create feature matrix
    X = df[features]
#     build model
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state = 42, n_init='auto'))
#     fit model
    model.fit(X)
    
#     return metrics
    if return_metrics:
#         calculate inertia
        i = model.named_steps["kmeans"].inertia_
#         calculate silhouette score
        ss = silhouette_score(X ,model.named_steps["kmeans"].labels_)
#         put results into dictionary
        metrics = {
            "inertia": round(i), 
            "silhoutte": round(ss,3)
        }
#         return metrics if return_metrics is false
        return metrics
#     return model if return_metrics is true
    return model


# K-means model and slider: connection layer
@callback(
    Output("metrics", "children"),
    Input("trim-button", "value"),
    Input("k-slider", "value")
)
def serve_metrics(trimmed = True, k = 2): # K-means model and slider: frontend

    """Returns list of ``H3`` elements containing inertia and silhouette score
    for ``KMeans`` model.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
#     get metrics
    metrics = get_model_metrics(trimmed = trimmed, k = k, return_metrics = True)
#     add metrics to html element
    text = [
        html.H3(f"Inertia: {metrics['inertia']}"),
        html.H3(f"Silhoutte Score: {metrics['silhoutte']}"),

    ]
    # return inertia ad silhoutte score metrics
    return text


# PCA scatterplot: backend
def get_pca_labels(trimmed=True, k=2):

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
#     create feature matrix
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    
#     build transformers
    transformer = PCA(n_components = 2, random_state = 42)
    
#     transform data
    X_t = transformer.fit_transform(X)
    X_pca = pd.DataFrame(X_t, columns = ["PC1", "PC2"])
    
#     add labels
    model = get_model_metrics(trimmed=trimmed, k=k, return_metrics=False)
    X_pca["labels"] = model.named_steps['kmeans'].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)
    
    return X_pca



# PCA scatterplot: connection layer
@callback(
    Output("pca-scatter","figure"),
    Input("trim-button", "value"),
    Input("k-slider", "value")
)
def serve_scatter_plot(trimmed=True, k=2): # PCA scatterplot: frontend


    """Build 2D scatter plot of ``df`` with ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    fig = px.scatter(
        data_frame=get_pca_labels(trimmed=trimmed, k=k),
        x = "PC1",
        y = "PC2",
        color = "labels",
        title = "PCA Representation of Clusters"
    )
    
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
