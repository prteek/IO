import streamlit as st
from src.projects import simulating_number_of_samples as snos
from src.projects import battery_model_parameter_estimation as bmpe
from src.projects import personal_polynomial as ppo
from src.projects import drive_cycle_characterisation as dcc
from src.projects import raspberry_pi_stats as rpis
from src.projects import hastie_monitoring as hm
from src.projects import london_bedroom_prediction as lbp
from src.projects import strava_project as sp


def run():
    
    class home_page():
        def __init__(self):
            return None

        def run(self):
            st.title('Projects home')
            st.markdown("### Welcome ! ")
            st.write("""This is a collection of short projects that I made as an effort in my Data Science journey.""")

    home = home_page()
    
    projects = {'Home':home,
                'Strava project':sp,
                # 'Hastie pipeline monitoring': hm,
                # 'London bedroom prediction':lbp,
                # 'Determining number of samples for a study':snos,
                'Drive cycle characterisation':dcc,
                # 'Battery model parameter estimation':bmpe,
#                 'Personal Polynomial':ppo,
#                 'Raspberry Pi Stats dashboard':rpis,
               }

    page = st.selectbox(label='Project list', options=list(projects.keys()))

    projects[page].run()