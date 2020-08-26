import streamlit as st
from src.blogs import chittr as chit
from src.blogs import relativity as rltv
from src.blogs import dillema_of_autonomous_vehicles as doav

def run():
    
    projects = {'Home':chit,
                'Logical notes on relativity':rltv,
                'Dillema of Autonomous Vehicles': doav,
               }

    page = st.sidebar.selectbox(label='Blogs', options=list(projects.keys()))

    projects[page].run()