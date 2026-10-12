"""Isolated staging page: validate here before changing the main dashboard."""
import streamlit as st
from data import synthetic_data, geographic_counts
from night_globe import render_globe
st.set_page_config(layout='wide')
st.title('Night globe prototype — synthetic data')
people, _ = synthetic_data()
render_globe(geographic_counts(people))
