"""Plotly presentation helpers; participant counts are supplied by data.py."""
import plotly.express as px
import plotly.graph_objects as go

def community_map(locations):
    fig = px.choropleth(locations, locations='iso3', color='Participants',
                        hover_name='country', color_continuous_scale='Blues',
                        template='plotly_white',
                        labels={'Participants': 'Synthetic participants'})
    # Scattergeo supports a limited symbol set; use circles instead of HTML icons.
    fig.add_trace(go.Scattergeo(
        lat=locations.latitude.tolist(), lon=locations.longitude.tolist(),
        mode='markers+text', text=locations.Participants.astype(str).tolist(),
        textposition=locations.label_position.tolist(),
        textfont=dict(size=10, color='#172554'),
        marker=dict(symbol='circle', size=5, color='#ffffff',
                    line=dict(color='#42658a', width=1)),
        customdata=locations[['country', 'Participants']].values.tolist(),
        hovertemplate='%{customdata[0]}<br>Synthetic participants: %{customdata[1]:,}<extra></extra>',
        name='Synthetic participant counts', showlegend=False,
    ))
    fig.update_traces(marker_line_color='#a6b4c4', marker_line_width=0.5,
                      selector=dict(type='choropleth'))
    fig.update_geos(
        projection_type='orthographic',
        showcoastlines=True, coastlinecolor='#a6b4c4', coastlinewidth=0.5,
        showcountries=True, countrycolor='#a6b4c4', countrywidth=0.5,
        showland=True, landcolor='#ffffff',
        showocean=True, oceancolor='#ffffff',
        showlakes=True, lakecolor='#ffffff',
        bgcolor='#ffffff', showframe=False,
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      font_color='#172554')
    return fig
