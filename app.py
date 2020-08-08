import streamlit as st
from src.projects import projects_home as ph
from src.blogs import blog_home as bh




class home_page():
    def __init__(self):
        return None
    
    def run(self):
        st.title('Home')
        st.write('This is home')


home = home_page()
    
projects = {'Home':home,
            'Projects':ph, 
         'Blog posts':bh}


page = st.sidebar.radio(label='Go to', options=list(projects.keys()))

projects[page].run()