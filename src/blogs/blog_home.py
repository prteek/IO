import streamlit as st
from src.blogs import chittr as chit

def run():
    
    projects = {'Home':chit,
               }

#     page = st.sidebar.selectbox(label='Blogs', options=list(projects.keys()))
    page = 'Home'
    projects[page].run()