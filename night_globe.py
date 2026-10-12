"""Offline Globe.GL iframe prototype. Only aggregate country records leave Python."""
import base64
import json
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).parent / 'assets' / 'globe'

def marker_records(locations):
    return [{'country': str(row.country), 'count': int(row.Participants),
             'lat': float(row.latitude), 'lng': float(row.longitude),
             'position': str(row.label_position)} for row in locations.itertuples()]

@lru_cache(maxsize=1)
def asset_content():
    script = (ASSETS / 'globe.gl.min.js').read_text(encoding='utf-8')
    texture = base64.b64encode((ASSETS / 'earth-night.jpg').read_bytes()).decode('ascii')
    return script, texture

def globe_html(locations):
    script, texture = asset_content()
    template = (ASSETS / 'globe.html').read_text(encoding='utf-8')
    # JSON is a data block, never executable text. Escape HTML parser delimiters.
    payload = json.dumps(marker_records(locations), ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return template.replace('__GLOBE_SCRIPT__', script).replace('__TEXTURE__', texture).replace('__MARKERS__', payload)

def render_globe(locations):
    import streamlit as st
    st.iframe(globe_html(locations), height=680, width='stretch',
              alt='Nighttime Earth with synthetic country participant counts. Drag to rotate; scroll to zoom.')
