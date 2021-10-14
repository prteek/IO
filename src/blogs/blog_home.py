import streamlit as st
from src.blogs import chittr as chit
from src.blogs import fit500 as fit500

def run():
    
    projects = {'Home':chit,
               'Fit 500 models on GPU':fit500}

    page = st.sidebar.selectbox(label='Blogs', options=list(projects.keys()))
    projects[page].run()