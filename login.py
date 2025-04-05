import designer
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader




def app():
     
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
        ) 
        
        c1,c2,c3=st.columns(3)
        option = c1.selectbox('',
                 ('Login', 'Register')
             )
        if option == 'Login':
            name, authentication_status, username = authenticator.login(':key: Login', 'main')
            
            if authentication_status:
                authenticator.logout('Logout', 'main')
                if username!='':
                    st.write(f'Welcome *{name}*')
                  
                    designer.app()
            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status == None:
                st.warning('Please enter your username and password')
                
        if option == 'Register':
            try:
                if authenticator.register_user(':bust_in_silhouette: Register',preauthorization=False):
                    st.success('User registered successfully')
            except Exception as e:
                st.error(e)
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False) 
        
        
        
        
       
        
        
        
        
        


                   

         
            
