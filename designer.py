import streamlit as st
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import uuid
import re
import joblib
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours, UtilityFunction
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL


def app():
    
    st.title('PCM-CF Designer')
   
    st.divider()
    
    st.markdown('### Instructions for use')
    st.markdown('##### _This web is pre-set based on the commonly used mixed heating temperature of :red[600 ℃] and heating time of :red[2 h] in existing research. If you want to design EG/molten salt composite PCM, please use EG with a mesh size of :red[50] and a pore size of :red[300 μm]._')
    st.divider()

    st.markdown('### Design of Molten Salt PCM and EG/Molten Salt Composite PCM')
    selected_option = st.selectbox('Please choose an option',['Molten salt PCM', 'EG/molten salt composite PCM'])
    
    if selected_option == 'Molten salt PCM':

        ETR_MP = joblib.load("ETR_MP.pkl")
        ETR_HF = joblib.load("ETR_HF.pkl")
        ETR_TC = joblib.load("ETR_TC.pkl")
        
        df=pd.read_excel("salts(EG=0).xlsx")
        salts_df = df.iloc[:,:30]
        elements_df = pd.read_excel("elements-特征.xlsx")
        element_features = {}
        for index, row in elements_df.iterrows():
            element_name = row['Element']
            element_features[element_name] = row.to_dict()

        X = pd.DataFrame()


        for index, row in salts_df.iterrows():
            feature_totals = {} 
            for element, amount in row.items():  
                if pd.notna(amount):  
                    element_features_dict = element_features.get(element)
                    if element_features_dict:  
                        for feature, feature_value in element_features_dict.items():
                            if feature != 'Element':  
                                feature_totals[feature] = feature_totals.get(feature, 0) + amount * feature_value * 0.01
        
            if not X.empty:
                X = pd.concat([X, pd.DataFrame([feature_totals], index=[index])], axis=0, ignore_index=True)
            else:
                X = pd.DataFrame([feature_totals], index=[index])
        X['EG'] = 0
        X['Mesh numbers'] = 0
        X['Aperture'] = 0
        X['Expansion rate'] = 0
        X['Thermal conductivity'] = 0
        X['T'] = df['T']
        X['t'] = df['t']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        MP = df.iloc[:,-3] 
        HF = df.iloc[:,-2] 
        TC = df.iloc[:,-1]
        

        def MP_HF_TC(composition_df):
            elements_df = pd.read_excel("elements-特征.xlsx")

            element_features = {}
            for index, row in elements_df.iterrows():
                element_name = row['Element']
                element_features[element_name] = row.to_dict()
   
            X = pd.DataFrame()

            for index, row in composition_df.iterrows():
                feature_totals = {} 
                for element, amount in row.items(): 
                    if pd.notna(amount):  
                        element_features_dict = element_features.get(element)
                        if element_features_dict:  
                            for feature, feature_value in element_features_dict.items():
                                if feature != 'Element': 
                                    feature_totals[feature] = feature_totals.get(feature, 0) + amount * feature_value * 0.01
                X = pd.DataFrame([feature_totals], index=[index])

            X['EG'] = 0
            X['Mesh numbers'] = 0
            X['Aperture'] = 0
            X['Expansion rate'] = 0
            X['Thermal conductivity'] = 0
            X['T'] = 600
            X['t'] = 2

            X_scaled = scaler.transform(X)

            MP =  np.exp(ETR_MP.predict(X_scaled.reshape(1, -1))[0])
            HF =  np.exp(ETR_HF.predict(X_scaled.reshape(1, -1))[0])
            TC =  np.exp(ETR_TC.predict(X_scaled.reshape(1, -1))[0])

            return (MP, HF, TC)
            
        mp_threshold = st.number_input("Enter the MP threshold value", max_value=800.0, step=50.0)
        hf_threshold = st.number_input("Enter the HF threshold value", min_value=100.0, step=50.0)
        st.divider()
        
        space = {
            'LiF':(0,81),'LiCl':(0,94),'LiBr':(0,68),'Li2SO4':(0,71),'Li2CO3':(0,62),'LiNO3':(0,97),'NaF':(0,75),'NaCl':(0,66),
            'NaBr':(0,73),'Na2SO4':(0,68),'NaNO3':(0,96),'Na2CO3':(0,60),'KF':(0,85),'KCl':(0,64),'KBr':(0,60),'K2SO4':(0,28),
            'KNO3':(0,90),'K2CO3':(0,72),'RbF':(0,84),'CsCl':(0,73),'MgF2':(0,33),'MgCl2':(0,68),'MgBr2':(0,55),'CaF2':(0,41),
            'CaCl2':(0,81),'Ca(NO3)2':(0,82),'SrCl2':(0,68),'Sr(NO3)2':(0,8),'BaCl2':(0,53),'Ba(NO3)2':(0,2)
              }

        charge_1 = ['LiF','LiCl','LiBr','Li2SO4','Li2CO3','LiNO3','NaF','NaCl','NaBr','Na2SO4','NaNO3','Na2CO3','KF','KCl','KBr',
           'K2SO4','KNO3','K2CO3','RbF','CsCl']
        charge_2 = ['MgF2','MgCl2','MgBr2','CaF2','CaCl2','Ca(NO3)2','SrCl2','Sr(NO3)2','BaCl2','Ba(NO3)2']

        exclusion_rules_1 = {
            'Li2SO4': ['BaCl2', 'Ba(NO3)2'],
            'Li2CO3': ['BaCl2', 'Ba(NO3)2', 'CaF2', 'CaCl2','Ca(NO3)2', 'MgF2', 'MgCl2', 'MgBr2'],
            'Na2CO3': ['BaCl2', 'Ba(NO3)2', 'CaF2', 'CaCl2','Ca(NO3)2', 'MgF2', 'MgCl2', 'MgBr2'],
            'Na2SO4': ['BaCl2', 'Ba(NO3)2'],
            'K2SO4': ['BaCl2', 'Ba(NO3)2'],
            'K2CO3': ['BaCl2', 'Ba(NO3)2', 'CaF2', 'CaCl2','Ca(NO3)2', 'MgF2', 'MgCl2', 'MgBr2']
                        }
        exclusion_rules_2 = {
                 'BaCl2': ['Ba(NO3)2', 'Li2SO4', 'Li2CO3', 'Na2CO3', 'Na2SO4', 'K2SO4', 'K2CO3'],
                 'CaF2': ['Li2CO3', 'Na2CO3', 'K2CO3'],
                 'CaCl2': ['Li2CO3', 'Na2CO3', 'K2CO3'],
                 'Ca(NO3)2': ['Li2CO3', 'Na2CO3', 'K2CO3'],
                 'MgF2': ['Li2CO3', 'Na2CO3', 'K2CO3'],
                 'MgCl2': ['Li2CO3', 'Na2CO3', 'K2CO3'],
                 'MgBr2': ['Li2CO3', 'Na2CO3', 'K2CO3']
                         }
        if st.button("Start Design"):
            results_12 = []
            def objective_function(**params):
                global all_results
                while True:
        
                    selected_salts_1 = np.random.choice(charge_1, replace=False)
                    selected_salts_2 = np.random.choice(charge_2, replace=False)
                    selected_salts = [selected_salts_1] + [selected_salts_2]
        
                    valid_combination = True
                    for salt1 in selected_salts:
                        if salt1 in exclusion_rules_1:
                            for salt2 in selected_salts:
                                if salt2 in exclusion_rules_1[salt1]:
                                    valid_combination = False
                                    break
                        if salt1 in exclusion_rules_2:
                            for salt2 in selected_salts:
                                if salt2 in exclusion_rules_2[salt1]:
                                    valid_combination = False
                                    break
                        if not valid_combination:
                            break
        
                    if valid_combination:
                        break
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
    
                if salt1_content + salt2_content != 100:
                    scale_factor = 100 / (salt1_content + salt2_content)
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
    
                mp, hf, tc = MP_HF_TC(composition_df)

                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_12.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
                   )

            converted_results_12 = []
            results_df_12 = pd.DataFrame(results_12)
        
            for index, row in results_df_12.iterrows():
                composition_series_12 = pd.Series(row['composition'])
                composition_series_12['MP'] = row['mp']
                composition_series_12['HF'] = row['hf']
                composition_series_12['TC'] = row['tc']
                composition_series_12['loss'] = row['loss']
    
                converted_results_12.append(composition_series_12)

            results_12 = pd.DataFrame(converted_results_12)
            min_loss_12 = results_12['loss'].min()
            min_loss_indices_12 = results_12[results_12['loss'] == min_loss_12].index
            results_12 = results_12.loc[min_loss_indices_12].reset_index(drop=True)
            results_12 = results_12.drop(columns=['loss'])

        
            results_11 = []
            def objective_function(**params):
                selected_salts = np.random.choice(charge_1, 2, replace=False)
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0 
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
    
                total_content = salt1_content + salt2_content
                if total_content != 100:
                    scale_factor = 100 / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
    
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_11.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                   )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
               )

            converted_results_11 = []
            results_df_11 = pd.DataFrame(results_11)
        
            for index, row in results_df_11.iterrows():
                composition_series_11 = pd.Series(row['composition'])
                composition_series_11['MP'] = row['mp']
                composition_series_11['HF'] = row['hf']
                composition_series_11['TC'] = row['tc']
                composition_series_11['loss'] = row['loss']
    
                converted_results_11.append(composition_series_11)

            results_11 = pd.DataFrame(converted_results_11)
            min_loss_11 = results_11['loss'].min()
            min_loss_indices_11 = results_11[results_11['loss'] == min_loss_11].index
            results_11 = results_11.loc[min_loss_indices_11].reset_index(drop=True)
            results_11 = results_11.drop(columns=['loss'])


            results_112 = []
            def objective_function(**params): 
                while True:
                    selected_class2_salt = np.random.choice(charge_2)
                    if selected_class2_salt in exclusion_rules_2:
                        valid_salt1 = [salt for salt in charge_1 if salt not in exclusion_rules_2[selected_class2_salt]]
                    else:
                        valid_salt1 = charge_1
        
                    selected_class1_salts = np.random.choice(valid_salt1, 2, replace=False)
        
                    valid_combination = True
                    for salt1 in selected_class1_salts:
                        if salt1 in exclusion_rules_1 and selected_class2_salt in exclusion_rules_1[salt1]:
                            valid_combination = False
                            break
                    if valid_combination:
                        break
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                salt1_content = np.random.uniform(space[selected_class1_salts[0]][0], space[selected_class1_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_class1_salts[1]][0], space[selected_class1_salts[1]][1])
                salt3_content = np.random.uniform(space[selected_class2_salt][0], space[selected_class2_salt][1])
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != 100:
                    scale_factor = 100 / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
    
                composition_df.loc[0, selected_class1_salts[0]] = salt1_content
                composition_df.loc[0, selected_class1_salts[1]] = salt2_content
                composition_df.loc[0, selected_class2_salt] = salt3_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_112.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
             n_iter=50,
            acquisition_function=utility
               )

            converted_results_112 = []
            results_df_112 = pd.DataFrame(results_112)
        
            for index, row in results_df_112.iterrows():
                composition_series_112 = pd.Series(row['composition'])
                composition_series_112['MP'] = row['mp']
                composition_series_112['HF'] = row['hf']
                composition_series_112['TC'] = row['tc']
                composition_series_112['loss'] = row['loss']
    
                converted_results_112.append(composition_series_112)

            results_112 = pd.DataFrame(converted_results_112)
            min_loss_112 = results_112['loss'].min()
            min_loss_indices_112 = results_112[results_112['loss'] == min_loss_112].index
            results_112 = results_112.loc[min_loss_indices_112].reset_index(drop=True)
            results_112 = results_112.drop(columns=['loss'])


            results_122 = []
            def objective_function(**params):
                while True:
                    selected_class1_salt = np.random.choice(charge_1)
                    if selected_class1_salt in exclusion_rules_1:
                        valid_salt2 = [salt for salt in charge_2 if salt not in exclusion_rules_1[selected_class1_salt]]
                    else:
                        valid_salt2 = charge_2
        
                    selected_class2_salts = np.random.choice(valid_salt2, 2, replace=False)
                    valid_combination = True
                    for salt2 in selected_class2_salts:
                        if salt2 in exclusion_rules_2 and selected_class1_salt in exclusion_rules_2[salt2]:
                            valid_combination = False
                            break
                    if valid_combination:
                        break
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                salt1_content = np.random.uniform(space[selected_class1_salt][0], space[selected_class1_salt][1])
                salt2_content = np.random.uniform(space[selected_class2_salts[0]][0], space[selected_class2_salts[0]][1])
                salt3_content = np.random.uniform(space[selected_class2_salts[1]][0], space[selected_class2_salts[1]][1])
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != 100:
                    scale_factor = 100 / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
    
                composition_df.loc[0, selected_class1_salt] = salt1_content
                composition_df.loc[0, selected_class2_salts[0]] = salt2_content
                composition_df.loc[0, selected_class2_salts[1]] = salt3_content
    
                mp, hf, tc = MP_HF_TC(composition_df)

                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_122.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                  )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
               )

            converted_results_122 = []
            results_df_122 = pd.DataFrame(results_122)
        
            for index, row in results_df_122.iterrows():
                composition_series_122 = pd.Series(row['composition'])
                composition_series_122['MP'] = row['mp']
                composition_series_122['HF'] = row['hf']
                composition_series_122['TC'] = row['tc']
                composition_series_122['loss'] = row['loss']
    
                converted_results_122.append(composition_series_122)

            results_122 = pd.DataFrame(converted_results_122)
            min_loss_122 = results_122['loss'].min()
            min_loss_indices_122 = results_122[results_122['loss'] == min_loss_122].index
            results_122 = results_122.loc[min_loss_indices_122].reset_index(drop=True)
            results_122 = results_122.drop(columns=['loss'])


            results_111 = []
            def objective_function(**params):
                selected_salts = np.random.choice(charge_1, 3, replace=False)
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0 
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
                salt3_content = np.random.uniform(space[selected_salts[2]][0], space[selected_salts[2]][1])
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != 100:
                    scale_factor = 100 / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
                
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
                composition_df.loc[0, selected_salts[2]] = salt3_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_111.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )
  
            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
                 )

            converted_results_111 = []
            results_df_111 = pd.DataFrame(results_111)
        
            for index, row in results_df_111.iterrows():
                composition_series_111 = pd.Series(row['composition'])
                composition_series_111['MP'] = row['mp']
                composition_series_111['HF'] = row['hf']
                composition_series_111['TC'] = row['tc']
                composition_series_111['loss'] = row['loss']
    
                converted_results_111.append(composition_series_111)

            results_111 = pd.DataFrame(converted_results_111)
            min_loss_111 = results_111['loss'].min()
            min_loss_indices_111 = results_111[results_111['loss'] == min_loss_111].index
            results_111 = results_111.loc[min_loss_indices_111].reset_index(drop=True)
            results_111 = results_111.drop(columns=['loss'])

            results_df = pd.concat([results_11,results_12,results_111,results_112,results_122], ignore_index=True)
            st.dataframe(results_df)


    
    else:
        ETR_MP = joblib.load("ETR_MP_EG.pkl")
        ETR_HF = joblib.load("ETR_HF_EG.pkl")
        ETR_TC = joblib.load("ETR_TC_EG.pkl")
        
        df=pd.read_excel("salts(EG!=0).xlsx")
        salts_df = df.iloc[:,:15]
        elements_df = pd.read_excel("elements-特征.xlsx")
        element_features = {}
        for index, row in elements_df.iterrows():
            element_name = row['Element']
            element_features[element_name] = row.to_dict()

        X = pd.DataFrame()


        for index, row in salts_df.iterrows():
            feature_totals = {} 
            for element, amount in row.items():  
                if pd.notna(amount):  
                    element_features_dict = element_features.get(element)
                    if element_features_dict:  
                        for feature, feature_value in element_features_dict.items():
                            if feature != 'Element':  
                                feature_totals[feature] = feature_totals.get(feature, 0) + amount * feature_value * 0.01
        
            if not X.empty:
                X = pd.concat([X, pd.DataFrame([feature_totals], index=[index])], axis=0, ignore_index=True)
            else:
                X = pd.DataFrame([feature_totals], index=[index])
        X['EG'] = salts_df['EG']
        X['Mesh numbers'] = df['Mesh numbers']
        X['Aperture'] = df['Aperture']
        X['Expansion rate'] = df['Expansion rate']
        X['Thermal conductivity'] = df['Thermal conductivity']
        X['T'] = df['T']
        X['t'] = df['t']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        MP = df.iloc[:,-3] 
        HF = df.iloc[:,-2] 
        TC = df.iloc[:,-1]
        

        def MP_HF_TC(composition_df):
            elements_df = pd.read_excel("elements-特征.xlsx")

            element_features = {}
            for index, row in elements_df.iterrows():
                element_name = row['Element']
                element_features[element_name] = row.to_dict()
   
            X = pd.DataFrame()

            for index, row in composition_df.iterrows():
                feature_totals = {} 
                for element, amount in row.items(): 
                    if pd.notna(amount):  
                        element_features_dict = element_features.get(element)
                        if element_features_dict:  
                            for feature, feature_value in element_features_dict.items():
                                if feature != 'Element': 
                                    feature_totals[feature] = feature_totals.get(feature, 0) + amount * feature_value * 0.01
                X = pd.DataFrame([feature_totals], index=[index])

            X['EG'] = salts_df['EG']
            X['Mesh numbers'] = df['Mesh numbers']
            X['Aperture'] = df['Aperture']
            X['Expansion rate'] = df['Expansion rate']
            X['Thermal conductivity'] = df['Thermal conductivity']
            X['T'] = df['T']
            X['t'] = df['t']

            X_scaled = scaler.transform(X)

            MP =  np.exp(ETR_MP.predict(X_scaled.reshape(1, -1))[0])
            HF =  np.exp(ETR_HF.predict(X_scaled.reshape(1, -1))[0])
            TC =  np.exp(ETR_TC.predict(X_scaled.reshape(1, -1))[0])

            return (MP, HF, TC)
            
        mp_threshold = st.number_input("Enter the MP threshold value", max_value=500.0, step=50.0)
        hf_threshold = st.number_input("Enter the HF threshold value", min_value=100.0, step=50.0)
        st.divider()
        
        space = {
             'LiCl':(0,45),'Li2CO3':(0,38),'LiNO3':(0,79),'NaF':(0,15),'NaCl':(0,51),'NaBr':(0,73),'Na2SO4':(0,48),
             'NaNO3':(0,86),'Na2CO3':(0,51),'KCl':(0,53),'KNO3':(0,85),'K2CO3':(0,34),'MgCl2':(5,50),
             'CaCl2':(0,47),'Ca(NO3)2':(0,76),'EG':(5,50)
                }

        charge_1 = ['LiCl','Li2CO3','LiNO3','NaF','NaCl','Na2SO4','NaNO3','Na2CO3','KCl','KNO3','K2CO3']
        charge_2 = ['MgCl2','CaCl2','Ca(NO3)2']

        exclusion_rules_1 = {
            'Li2CO3': [ 'CaCl2','Ca(NO3)2', 'MgCl2'],'Na2CO3': ['CaCl2','Ca(NO3)2',  'MgCl2'],
            'K2CO3': ['CaCl2','Ca(NO3)2',  'MgCl2']
                        }
        exclusion_rules_2 = {
                 'CaCl2': [ 'Li2CO3','Na2CO3', 'K2CO3'],'Ca(NO3)2': [ 'Li2CO3','Na2CO3', 'K2CO3'],
                 'MgCl2': [ 'Li2CO3','Na2CO3', 'K2CO3']
                         }
        if st.button("Start Design"):
            results_12 = []
            def objective_function(**params):
                global all_results
                while True:
        
                    selected_salts_1 = np.random.choice(charge_1, replace=False)
                    selected_salts_2 = np.random.choice(charge_2, replace=False)
                    selected_salts = [selected_salts_1] + [selected_salts_2]
        
                    valid_combination = True
                    for salt1 in selected_salts:
                        if salt1 in exclusion_rules_1:
                            for salt2 in selected_salts:
                                if salt2 in exclusion_rules_1[salt1]:
                                    valid_combination = False
                                    break
                        if salt1 in exclusion_rules_2:
                            for salt2 in selected_salts:
                                if salt2 in exclusion_rules_2[salt1]:
                                    valid_combination = False
                                    break
                        if not valid_combination:
                            break
        
                    if valid_combination:
                        break
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
                eg_content = np.random.uniform(space['EG'][0], space['EG'][1])
                salt_content = 100 - eg_content
    
                if salt1_content + salt2_content != salt_content:
                    scale_factor = salt_content / (salt1_content + salt2_content)
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
                composition_df.loc[0, 'EG'] = eg_content
    
                mp, hf, tc = MP_HF_TC(composition_df)

                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_12.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
                   )

            converted_results_12 = []
            results_df_12 = pd.DataFrame(results_12)
        
            for index, row in results_df_12.iterrows():
                composition_series_12 = pd.Series(row['composition'])
                composition_series_12['MP'] = row['mp']
                composition_series_12['HF'] = row['hf']
                composition_series_12['TC'] = row['tc']
                composition_series_12['loss'] = row['loss']
    
                converted_results_12.append(composition_series_12)

            results_12 = pd.DataFrame(converted_results_12)
            min_loss_12 = results_12['loss'].min()
            min_loss_indices_12 = results_12[results_12['loss'] == min_loss_12].index
            results_12 = results_12.loc[min_loss_indices_12].reset_index(drop=True)
            results_12 = results_12.drop(columns=['loss'])

        
            results_11 = []
            def objective_function(**params):
                selected_salts = np.random.choice(charge_1, 2, replace=False)
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0 
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
                eg_content = np.random.uniform(space['EG'][0], space['EG'][1])
                salt_content = 100 - eg_content
    
                if salt1_content + salt2_content != salt_content:
                    scale_factor = salt_content / (salt1_content + salt2_content)
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
    
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
                composition_df.loc[0, 'EG'] = eg_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_11.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                   )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
               )

            converted_results_11 = []
            results_df_11 = pd.DataFrame(results_11)
        
            for index, row in results_df_11.iterrows():
                composition_series_11 = pd.Series(row['composition'])
                composition_series_11['MP'] = row['mp']
                composition_series_11['HF'] = row['hf']
                composition_series_11['TC'] = row['tc']
                composition_series_11['loss'] = row['loss']
    
                converted_results_11.append(composition_series_11)

            results_11 = pd.DataFrame(converted_results_11)
            min_loss_11 = results_11['loss'].min()
            min_loss_indices_11 = results_11[results_11['loss'] == min_loss_11].index
            results_11 = results_11.loc[min_loss_indices_11].reset_index(drop=True)
            results_11 = results_11.drop(columns=['loss'])


            results_112 = []
            def objective_function(**params): 
                while True:
                    selected_class2_salt = np.random.choice(charge_2)
                    if selected_class2_salt in exclusion_rules_2:
                        valid_salt1 = [salt for salt in charge_1 if salt not in exclusion_rules_2[selected_class2_salt]]
                    else:
                        valid_salt1 = charge_1
        
                    selected_class1_salts = np.random.choice(valid_salt1, 2, replace=False)
        
                    valid_combination = True
                    for salt1 in selected_class1_salts:
                        if salt1 in exclusion_rules_1 and selected_class2_salt in exclusion_rules_1[salt1]:
                            valid_combination = False
                            break
                    if valid_combination:
                        break
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                salt1_content = np.random.uniform(space[selected_class1_salts[0]][0], space[selected_class1_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_class1_salts[1]][0], space[selected_class1_salts[1]][1])
                salt3_content = np.random.uniform(space[selected_class2_salt][0], space[selected_class2_salt][1])
                eg_content = np.random.uniform(space['EG'][0], space['EG'][1])
                salt_content = 100 - eg_content
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != salt_content:
                    scale_factor = salt_content / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
    
                composition_df.loc[0, selected_class1_salts[0]] = salt1_content
                composition_df.loc[0, selected_class1_salts[1]] = salt2_content
                composition_df.loc[0, selected_class2_salt] = salt3_content
                composition_df.loc[0, 'EG'] = eg_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_112.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
             n_iter=50,
            acquisition_function=utility
               )

            converted_results_112 = []
            results_df_112 = pd.DataFrame(results_112)
        
            for index, row in results_df_112.iterrows():
                composition_series_112 = pd.Series(row['composition'])
                composition_series_112['MP'] = row['mp']
                composition_series_112['HF'] = row['hf']
                composition_series_112['TC'] = row['tc']
                composition_series_112['loss'] = row['loss']
    
                converted_results_112.append(composition_series_112)

            results_112 = pd.DataFrame(converted_results_112)
            min_loss_112 = results_112['loss'].min()
            min_loss_indices_112 = results_112[results_112['loss'] == min_loss_112].index
            results_112 = results_112.loc[min_loss_indices_112].reset_index(drop=True)
            results_112 = results_112.drop(columns=['loss'])


            results_122 = []
            def objective_function(**params):
                while True:
                    selected_class2_salts = np.random.choice(charge_2, 2, replace=False)
                    excluded_salt1 = set()
                    for salt in selected_class2_salts:
                        if salt in exclusion_rules_2:
                            excluded_salt1.update(exclusion_rules_2[salt])
                    valid_salt1 = [salt for salt in charge_1 if salt not in excluded_salt1]
                    selected_class1_salt = np.random.choice(valid_salt1, replace=False)
                    valid_combination = True
                    for salt2 in selected_class2_salts:
                        if salt2 in exclusion_rules_2 and selected_class1_salt in exclusion_rules_2[salt2]:
                            valid_combination = False
                            break
                    if valid_combination:
                        break
    
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0  
    
                salt1_content = np.random.uniform(space[selected_class1_salt][0], space[selected_class1_salt][1])
                salt2_content = np.random.uniform(space[selected_class2_salts[0]][0], space[selected_class2_salts[0]][1])
                salt3_content = np.random.uniform(space[selected_class2_salts[1]][0], space[selected_class2_salts[1]][1])
                eg_content = np.random.uniform(space['EG'][0], space['EG'][1])
                salt_content = 100 - eg_content
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != salt_content:
                    scale_factor = salt_content / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
    
                composition_df.loc[0, selected_class1_salt] = salt1_content
                composition_df.loc[0, selected_class2_salts[0]] = salt2_content
                composition_df.loc[0, selected_class2_salts[1]] = salt3_content
                composition_df.loc[0, 'EG'] = eg_content
    
                mp, hf, tc = MP_HF_TC(composition_df)

                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_122.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )

            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                  )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
               )

            converted_results_122 = []
            results_df_122 = pd.DataFrame(results_122)
        
            for index, row in results_df_122.iterrows():
                composition_series_122 = pd.Series(row['composition'])
                composition_series_122['MP'] = row['mp']
                composition_series_122['HF'] = row['hf']
                composition_series_122['TC'] = row['tc']
                composition_series_122['loss'] = row['loss']
    
                converted_results_122.append(composition_series_122)

            results_122 = pd.DataFrame(converted_results_122)
            min_loss_122 = results_122['loss'].min()
            min_loss_indices_122 = results_122[results_122['loss'] == min_loss_122].index
            results_122 = results_122.loc[min_loss_indices_122].reset_index(drop=True)
            results_122 = results_122.drop(columns=['loss'])


            results_111 = []
            def objective_function(**params):
                selected_salts = np.random.choice(charge_1, 3, replace=False)
                composition_df = pd.DataFrame(columns=salts_df.columns)
                composition_df.loc[0] = 0 
    
                salt1_content = np.random.uniform(space[selected_salts[0]][0], space[selected_salts[0]][1])
                salt2_content = np.random.uniform(space[selected_salts[1]][0], space[selected_salts[1]][1])
                salt3_content = np.random.uniform(space[selected_salts[2]][0], space[selected_salts[2]][1])
                eg_content = np.random.uniform(space['EG'][0], space['EG'][1])
                salt_content = 100 - eg_content
    
                total_content = salt1_content + salt2_content + salt3_content
                if total_content != salt_content:
                    scale_factor = salt_content / total_content
                    salt1_content *= scale_factor
                    salt2_content *= scale_factor
                    salt3_content *= scale_factor
                
                composition_df.loc[0, selected_salts[0]] = salt1_content
                composition_df.loc[0, selected_salts[1]] = salt2_content
                composition_df.loc[0, selected_salts[2]] = salt3_content
                composition_df.loc[0, 'EG'] = eg_content
    
                mp, hf, tc = MP_HF_TC(composition_df)
         
                mp_norm = (mp - np.min(MP)) / (np.max(MP) - np.min(MP))
                hf_norm = (hf - np.min(HF)) / (np.max(HF) - np.min(HF))
                tc_norm = (tc - np.min(TC)) / (np.max(TC) - np.min(TC))

                w1, w2, w3 = 0.3, 3, 1.5
                loss = w1 * mp_norm - w2 * hf_norm - w3 * tc_norm + 100 * (hf < hf_threshold) + 100 * (mp > mp_threshold)
    
                result = {
                  'loss': loss,  
                   'mp': mp,
                   'hf': hf,
                   'tc':tc,
                   'status': STATUS_OK,
                   'composition': composition_df.iloc[0].to_dict()  
                    }
                results_111.append({'composition': composition_df.iloc[0].to_dict(), 'mp': mp, 'hf': hf, 'tc': tc, 'loss':loss})
                return result['loss']

            optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=space,
            random_state=10,
            verbose=1 
               )
  
            optimizer.set_gp_params(
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b" 
                 )

            utility = UtilityFunction(kind="ei")

            optimizer.maximize(
            init_points=100,
            n_iter=50,
            acquisition_function=utility
                 )

            converted_results_111 = []
            results_df_111 = pd.DataFrame(results_111)
        
            for index, row in results_df_111.iterrows():
                composition_series_111 = pd.Series(row['composition'])
                composition_series_111['MP'] = row['mp']
                composition_series_111['HF'] = row['hf']
                composition_series_111['TC'] = row['tc']
                composition_series_111['loss'] = row['loss']
    
                converted_results_111.append(composition_series_111)

            results_111 = pd.DataFrame(converted_results_111)
            min_loss_111 = results_111['loss'].min()
            min_loss_indices_111 = results_111[results_111['loss'] == min_loss_111].index
            results_111 = results_111.loc[min_loss_indices_111].reset_index(drop=True)
            results_111 = results_111.drop(columns=['loss'])

            results_df = pd.concat([results_11,results_12,results_111,results_112,results_122], ignore_index=True)
            st.dataframe(results_df)

            



        

        



        

        


        

        
    
        
        
    

        

    
    





