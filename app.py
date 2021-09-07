import streamlit as st
from src.projects import projects_home as ph
from src.blogs import blog_home as bh




class home_page():
    def __init__(self):
        return None
    
    def run(self):
        st.title('IO')
        st.write("""Hi ! I am your just another friendly neighbourhood Data-Scientist.  
Similar to your other friendly neighbourhood costumed crusaders, I believe that "with great power comes great responsibility".  
That is why I rely on sound logic and principles of physics and math for designing and analysing stuff that matter.  
In the past I have worked with Automotive companies like:
* Suzuki
* Jaguar Land Rover
* Ford
* Allison Transmission

To help them make sense of the customer usage (and powertrain systems) data that they have, to aid in reasearch and development of new products, particularly Hybrid Electric vehicles.  

I have deep interest in Statistics, Machine learning and Python and I occassionally dabble in Astrophysics and Photography.  

If you'd like to get in touch, please use the details below:


        """)
        
        st.markdown(""" ###  
---  
[Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
[Repository](https://github.com/prteek/IO/ "Github")

    """)


home = home_page()
    
projects = {'Home':home,
            'Projects':ph,
            'Chittr':bh,
           }


page = st.sidebar.radio(label='Go to', options=list(projects.keys()))

projects[page].run()