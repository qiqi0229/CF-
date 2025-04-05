import streamlit as st
import home, usage, login, contact
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="PCM-CF",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
   
)


st.sidebar.title('Welcome to our website! :sunglasses:')
with st.sidebar:
    app = option_menu(
            menu_title='Menu',
            options=['About','Users Guide','PCM-CF Designer','Contact'],
            icons=['house-fill','info-circle-fill','person-circle','chat-fill'],
            menu_icon='chat-text-fill',
            default_index=0,
            styles={
                    "container": {"padding": "5!important","background-color":'grey'},
                    "icon": {"color": "white", "font-size": "23px"}, 
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"}}
                        )

if app == 'About':
    home.app() 
    
if app == 'PCM-CF Designer':
    login.app()
                   
if app == "Users Guide":
    usage.app()
        
if app == 'Contact':
    contact.app()


