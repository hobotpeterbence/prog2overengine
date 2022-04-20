import dash
import dask.bag as db
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
from dask_ml.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title = "apppróba"


app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            id="H1",
            children="Írj be egy cikket kapsz vissza egy hasonló G7-est!",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        dcc.Input(
            id="input_text",
            type="text",
            placeholder="input type text",
            style=dict(display="flex", justifyContent="center"),
        ),
    ]
    + [html.Div(id="out-all-types")],
)


@app.callback(
    Output(component_id="out-all-types", component_property="children"),
    Input(component_id="input_text", component_property="value"),
)
def process_tfidf_similarity(basedoc):
    basedoc = str(basedoc)
    basedoc = basedoc.replace(",", "").replace(".", "").replace("”", "")
    basedoc = basedoc.lower()
    data = pd.read_csv("g7_kicsi.csv", encoding="UTF-32")
    data.columns = ["index1", "index2", "link", "szoveg"]
    documents = list(data["szoveg"])

    vectorizer = CountVectorizer()

    # To make uniformed vectors, both documents need to be combined first.
    documents.insert(0, basedoc)
    corpus = db.from_sequence(documents, npartitions=2)
    embeddings = vectorizer.fit_transform(corpus)
    embeddings = embeddings.compute().toarray()
    cosine_sim = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()

    most_similar_document = documents[cosine_sim.argmax() + 1]

    return cosine_sim.max(), most_similar_document


if __name__ == "__main__":
    app.run_server(debug=True)
