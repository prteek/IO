# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Home page for projects 

import streamlit as st
from PIL import Image

st.title("IO Python projects")

"""### Welcome !

This is a collection of short fun (and not so fun) projects.  
Each project is exercise in learning python and expanding my understanding of Data analysis.
#
"""

image = Image.open("./docs/IO.jpg")
st.image(image, caption="IO: Third largest moon of Jupyter", use_column_width=True)

toc_body = """
# IO

[Sample size estimation for survival analysis](http://battery-failure-study.herokuapp.com)  
[Personal Polynomial](https://personal-polynomial.herokuapp.com)  
[Raspberry Pi Dashboard](https://prateek-rpi4-stats.anvil.app)  
[COVID19 early stage forecast](https://covid-19-early-stage-forecast.herokuapp.com)  
[Battery modelling](https://battery-modelling.herokuapp.com)  
[Twitter Data Analysis](https://github.com/prteek/IO/tree/fetch_twitter_data)  
[World Bank Data Analysis](https://github.com/prteek/IO/tree/world_bank_data_plots)  
[Document Scanner](https://github.com/prteek/IO/tree/scanner-app)  
[Bayesian reasoning](https://github.com/prteek/IO/tree/bayesian_reasoning)  
[Drive cycle for fuel economy](https://github.com/prteek/IO/tree/drive_cycles_for_fuel_economy)  
[Dash coin mining contract analysis](https://github.com/prteek/IO/tree/dash_coin_mining_analysis)  
[iOS shortcut query twitter](https://www.icloud.com/shortcuts/1ff69749a4b34a93b2872d2d6014755f)  
[iOS shortcut Photo OCR](https://www.icloud.com/shortcuts/8671db5505c940b68fcf61fb7eff5757)  
[iOS shortcut Morning Routine](https://www.icloud.com/shortcuts/bdb862fe3645495d935ee9d082239dc1)  

"""
st.sidebar.markdown(toc_body)


""" ###
---
[Prateek](mailto:prateekpatel.in@gmail.com "Email")  
[Repository](https://github.com/prteek/IO/tree/master "Github")
"""
