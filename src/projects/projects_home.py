import streamlit as st
from src.projects import simulating_number_of_samples as snos
from src.projects import battery_model_parameter_estimation as bmpe
from src.projects import personal_polynomial as ppo
from src.projects import coronavirus_eda_and_prediction as ceap
from src.projects import drive_cycle_characterisation as dcc

def run():
    
    class home_page():
        def __init__(self):
            return None

        def run(self):
            st.title('Projects home')
            st.markdown("### Welcome ! ")
            st.write("""This is a collection of short fun (and not so fun) projects.
            Each project is exercise in learning python and expanding
            my understanding of Data analysis.""")

    home = home_page()
    
    projects = {'Home':home,
                'Determining number of samples for a study':snos, 
                'Battery model parameter estimation':bmpe,
                'Personal Polynomial':ppo,
                'COVID19 EDA and early stage model':ceap,
                'Drive cycle characterisation':dcc,
               }

    page = st.sidebar.selectbox(label='Projects', options=list(projects.keys()))

    projects[page].run()