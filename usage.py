import streamlit as st



def app():
    st.sidebar.info('Please carefully review the usage instructions!')
    st.title('Users Guide')
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['#### Step :one:', "#### Step :two:", "#### Step :three:", '#### Step :four:', '#### Step :five:'])
    with tab1:
        st.markdown('#### 1. Firstly, log in and enter your username and password. If you do not have an account, click on the dropdown menu at the top of the page and select "Register".')
        st.image(
            'Usage1.png'
            ,width=900
        )
    
    with tab2:
        st.markdown('#### 2. Pay attention to reading the instructions here, which explain the synthesis process and EG parameters.')
        st.image(
            'Usage2.png'
            ,width=900
        )
        
    with tab3:
        st.markdown('#### 3. Select the type of molten salt you want to design from the drop-down menu.')
        st.image(
            'Usage3.png'
            ,width=900
        )
    with tab4:
        st.markdown('#### 4. Enter the upper limit value of MP and the lower limit value of HF for the PCM that you want to design.')
        st.image(
            'Usage4.png'
            ,width=900
        )
        
    with tab5:
        st.markdown('#### 5. Click the "Start Design" button to initiate the  design process and obtain the formula design results.')
        st.image(
            'Usage5.png'
            ,width=900
        )
      

