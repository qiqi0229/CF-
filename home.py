import streamlit as st
import pandas as pd
import re
import uuid
import json



def app():

    
    st.sidebar.warning('#### Version\n 1.1.0\n #### Last Updated\n 2025-02-12')
    st.title('PCM-CF: Composition Formula Designer for Multicomponent Molten Salts Phase Change Materials')
   
    st.divider()
    
    st.subheader(':book: Introduction')
    st.write('PCM-CF is a web app framework specifically designed for designing formulations of multicomponent molten salt phase change materials (PCM). The PCM-CF database primarily focuses on multi-component inorganic molten salts, such as LiF LiCl、LiBr、Li2SO4、Li2CO3、LiNO3、NaF、NaCl、NaBr、Na2SO4、NaNO3、Na2CO3、KF、KCl、KNO3 and more. It will provide comprehensive information on the composition, content, supporting material parameters, preparation process, and key performance indicators of PCM for molten salts. This PCM-CF application enables users to access a design tool for the purpose of designing novel materials. ')
    st.subheader(':exclamation: :red[Note]')
    st.markdown('Only by logging in to your account can you view the :red[**PCM-CF Designer**] pages!')
    st.divider()
    
    st.subheader(':mag: Database Overview')
    c1, c2, c3, c4 = st.columns(4)
    c1.info('##### PCMs\n 2')
    c2.info('##### Salts\n 30')
    c3.info('##### Records\n 574')
    c4.info('##### Literatures\n 48')
    st.write('\n- The database includes two categories: molten salt PCM and EG/molten salt composite PCM. It covers 30 common salt species, primarily categorized into five major classes: nitrate, carbonate, sulfate, and halide (mainly chloride salt and fluoride salt).')
    
    co1, co2 = st.columns(2)
    if co1.button('Molten salt PCM'):
        df1=pd.read_excel('Molten_salt_PCM.xlsx')
        st.dataframe(df1)
    if co2.button('EG/molten salt composite PCM'):
        df2=pd.read_excel('EG_molten_salt_composite_PCM.xlsx')
        st.dataframe(df2)
    
    st.divider()
    
    st.subheader(':bulb: Composition Formula Designer')
    st.write('Leveraging the PCM-CF database, we have employed advanced machine learning techniques to construct a formula design model. This model is specifically designed to provide precise design of the composition of multi-component molten salt phase change materials. Rigorous validation procedures have been conducted to guarantee the accuracy and robustness of the model. With this tool, users can participate in online formula design and optimization, enabling them to develop new materials based on performance requirements.\n The predictive models provide insights into the following key thermodynamic properties:\n\n- Melting Point(MP, ℃): The temperature at which substances transition from a solid to a liquid state.\n- Heat of Fusion(HF, J/g): The amount of heat absorbed by a unit mass of crystalline material to convert it into a liquid state at its melting point.\n- Thermal Conductivity(TC, W/m·K): A measure of the ability to conduct heat smoothly and effectively.')





    