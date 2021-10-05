import streamlit as st 
import pandas as pd
import joblib 
import os
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# DB Management
import sqlite3
conn = sqlite3.connect('users.db')
c = conn.cursor()


def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(firstname Text,lastname Text,username Text,password Text)')


def add_userdata(firstname,lastname,username,password):
	c.execute('INSERT INTO userstable(firstname,lastname,username,password) VALUES (?,?,?,?)',(firstname,lastname,username,password,))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def  view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def load_model(model_file):
	load_model = joblib.load(open(os.path.join(model_file),"rb"))
	return load_model

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key


def main():


	'''Obesity Detection APP'''
	 
	st.title('Obesity Prediction App')

	hide_menu_style = '''
		   <style>
		   #MainMenu{}
		   footer {visibility:hidden;}
		   </style>
		   '''
	st.markdown(hide_menu_style , unsafe_allow_html=True)


	menu = ['Home','Login','SignUp','Manage Users']
	choice = st.sidebar.selectbox('Menu',menu)
	 
	if choice == 'Home':

		if st.checkbox("Home"):

		   st.write('''*....Welcome to our Clinical Decision Support System Project....*''')
		   

		if st.checkbox('About'):

		   st.write('''This system helps the users at home and also clinics to predict whether a patient is a victim of Obesity or not after logging in or creating an account and then login if they dont have an account.''')
		   st.write('''The system requires a patient to enter Age, Weight, Height and Gender inorder to predict whether the patient is a victim of:''')
		   st.write('''1.Normal_Weight''')
		   st.write('''2.Overweight_Level_I''')
		   st.write('''3.Overweight_Level_II''')
		   st.write('''4.Obesity_Type_I''')
		   st.write('''5.Obesity_Type_II''')
		   st.write('''6.Obesity_Type_III''')


		   st.write('''The system also offers you the option to choose the Machine Model to use when predicting as these models vary in accuracy.''')
		   st.write('''The Machine Learning Models include:''')
		   st.write('''i.Logistic Regression(LR)''')
		   st.write('''ii.Random Forest Classifier(RFC)''')
		   st.write('''iii.Decision Tree Classifier(DTC)''')
		   st.write('''iv.KNearest Neighbors(KNN)''')


		   st.write('''The user can either plot and perform a prediction by selecting the following options respectively after logging in:''')
		   st.write('''a.Plot''')
		   st.write('''b.Prediction''')


	elif choice == 'Login':
		

		username = st.sidebar.text_input('username')
		password = st.sidebar.text_input('password',type='password')

		if st.sidebar.button('Login'):
			create_table()
			result = login_user(username,password)
            
            if result:

				st.success('Logged as {}'.format(username))

				st.subheader('***Prediction***')
			
				gender_dict = {'Male':1 ,'Female':0}

				st.write('''
					Enter the required values
					''')
				age = st.number_input('Enter Age',1,61)
				gender = st.radio("Gender",tuple(gender_dict.keys()))
				weight = st.number_input('Enter your Weight',1.0,173.0)
				height = st.number_input('Enter your Height',1.0,1.98)

				feature_list = [age,get_value(gender,gender_dict),weight,height]

				st.write(feature_list)

				pretty_result = {"age":age ,"gender":gender,"weight":weight,"height":height}

				st.json(pretty_result)

				sample_input = np.array(feature_list).reshape(1,-1)


				model_choice = st.selectbox("[Select Model]",["LR","DTC","KNN","RFC"])


				if st.button('Predict'):

					if model_choice == "LR":

						classifier = load_model('logistic_regression_obesity.pkl')
					
						prediction = classifier.predict(sample_input)

					elif model_choice == "DTC":
						classifier = load_model('decision_tree_classifier_obesity.pkl')
					
						prediction = classifier.predict(sample_input)

					
					elif model_choice == "KNN":
						classifier = load_model('knn_obesity.pkl')
					
						prediction = classifier.predict(sample_input)

						
					elif model_choice == "RFC":
						classifier = load_model('random_forest_classifier_obesity.pkl')
					
						prediction = classifier.predict(sample_input)
						

					st.write(prediction)
					
					if prediction == 0:
						st.success('Normal_Weight')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Mantain your Life style*''')
						st.write('''*2.Keep Exercising*''')

					elif prediction == 1:
						st.warning('Overweight_Level_I')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Exercise Daily*''')
						st.write('''*2.Eat Proper Diet*''')
						st.write('''*3.Avoid eating sugary foods*''')

						st.subheader("Medical Management")
						st.write('''*1.Consult a Doctor*''')
						st.write('''*2.Go for Checkups*''')

					elif prediction == 2:
						st.warning('Overweight_Level_II')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Exercise Daily*''')
						st.write('''*2.Eat Proper Diet*''')
						st.write('''*3.Avoid eating sugary foods*''')

						st.subheader("Medical Management")
						st.write('''*1.Consult a Doctor*''')
						st.write('''*2.Go for Checkups*''')


					elif prediction == 3:
						st.warning('Obesity_Type_I')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Exercise Daily*''')
						st.write('''*2.Eat Proper Diet*''')
						st.write('''*3.Avoid eating sugary foods*''')

						st.subheader("Medical Management")
						st.write('''*1.Consult a Doctor*''')
						st.write('''*2.Go for Checkups*''')

						
					elif prediction == 4:
						st.warning('Obesity_Type_III')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Exercise Daily*''')
						st.write('''*2.Eat Proper Diet*''')
						st.write('''*3.Avoid eating sugary foods*''')

						st.subheader("Medical Management")
						st.write('''*1.Consult a Doctor*''')
						st.write('''*2.Go for Checkups*''')

					elif prediction == 5:
						st.warning('Obesity_Type_II')
						st.header("Prescriptive Analysis")
						st.subheader("Recommended Life style Modification")
						st.write('''*1.Exercise Daily*''')
						st.write('''*2.Eat Proper Diet*''')
						st.write('''*3.Avoid eating sugary foods*''')
						
						st.subheader("Medical Management")
						st.write('''*1.Consult a Doctor*''')
						st.write('''*2.Go for Checkups*''')


			else:
				st.warning('Incorrect username or password')


	elif choice == 'SignUp':
		st.subheader('Create New Account')
		
		new_firstnme = st.text_input('firstname')
		new_lastnme = st.text_input('lastname')
		new_user = st.text_input('username')
		new_pasword = st.text_input('password',type='password')
		confirm_password = st.text_input("confirm_password",type ='password')
			

		if st.button('Signup'):

			if new_pasword == confirm_password :
				st.success('Password Confirmed')

			else:
				st.warning('Incorrect Password')
			
			create_table()
			add_userdata(new_firstnme,new_lastnme,new_user,new_pasword)
			st.success('You have successfully created an account')
			st.info('Go to Login menu to login')


	elif choice == 'Manage Users':
		st.subheader('Admin Login')
		username = st.text_input('username')
		password = st.text_input('password',type='password')

		if st.button('Login'):
				if password == 'admin123':
				   st.success("Welcome {}".format(username))
				
				   users_data = view_all_users()
				   users_db = pd.DataFrame(users_data,columns = ["firstname","lastname","username","password"])
				   st.dataframe(users_db)


				else:
					st.warning("Incorrect Username or Password")


if __name__ == '__main__':
	main()