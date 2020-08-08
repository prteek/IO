import streamlit as st
from src.blogs import blog_1 as b1
from src.blogs import blog_2 as b2

def run():
    
    class home_page():
        def __init__(self):
            return None

        def run(self):
            st.title('Blog home')
            st.write('Feeling Chatty')

    home = home_page()
    
    projects = {'Home':home,
        'blog 1':b1, 
     'blog 2':b2}

    page = st.sidebar.selectbox(label='Blogs', options=list(projects.keys()))

    projects[page].run()