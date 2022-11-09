import plotly.graph_objects as go

# Colorblind friendly palette
color_discrete_sequence = [
    "#648fff",
    "#785ef0",
    "#dc267f",
    "#fe6100",
    "#ffb000",
    "#000000",
]

fig_template = dict(
    layout=go.Layout(
        template="plotly_white",
        font_family="Arial",
        font_size=10,
        xaxis_showgrid=False,
        xaxis_ticks="inside",
        xaxis_mirror="allticks",
        xaxis_zeroline=False,
        xaxis_showline=True,
        xaxis_linecolor="black",
        yaxis_showgrid=False,
        yaxis_ticks="inside",
        yaxis_mirror="allticks",
        yaxis_zeroline=False,
        yaxis_showline=True,
        yaxis_linecolor="black",
        colorway=color_discrete_sequence,
    )
)

tag_symbols = dict(
    B1="circle",
    B2="star-triangle-up",
    B3="star-triangle-down",
    B4="star-square",
    H1="star-diamond",
    H2="diamond-tall",
    H3="diamond",
    H4="pentagon",
    H5="x",
    H6="hexagon",
)
