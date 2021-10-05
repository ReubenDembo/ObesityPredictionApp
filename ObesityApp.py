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
conn = sqlite3.connect('data.db')
c = conn.cursor()


def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(first_name Text,surname Text ,email Varchar(25),username Text,password Text,confirm_pasword Text)')


def add_userdata(first_name,surname,username,email,password):
	c.execute('INSERT INTO userstable(first_name,surname ,username,email,password) VALUES (?,?)',(first_name,surname,username,email,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE first_name = ? AND surname = ? AND username = ? AND email = ? AND password = ?',(first_name,surname,username,email,password))

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
		   footer {visibility:hidden;}
		   </style>
		   '''
	st.markdown(hide_menu_style , unsafe_allow_html=True)

	@st.cache(allow_output_mutation=True)
	def get_base64_of_bin_file(bin_file):
		with open(bin_file, 'rb') as f:
			data = f.read()
		return base64.b64encode(data).decode()

	def set_png_as_page_bg(jpg_file):
		bin_str = get_base64_of_bin_file(jpg_file)
		page_bg_img = '''
		<style>
		body {
		background-image: url("data:image/png;base64,%s");
		background-size: cover;
		}
		</style>
		''' % bin_str
		
		st.markdown(page_bg_img, unsafe_allow_html=True)
		return

	set_png_as_page_bg('pic.jpg')

	st.markdown(
	"""
	<style>
	.reportview-container .markdown-text-container {
		font-family: monospace;
	}
	.sidebar .sidebar-content {
		background-image: linear-gradient(#2e7bcf,#2e7bcf);
		color: white;
	}
	.Widget>label {
		color: white;
		font-family: monospace;
	}
	[class^="st-b"]  {
		color: white;
		font-family: monospace;
	}
	.st-at {
		background-color: #0c0080;
	}

	</style>
	""",
	unsafe_allow_html=True,
	)
	

	menu = ['Home','Login','SignUp','Manage Users']
	choice = st.sidebar.selectbox('Menu',menu)
	 
	if choice == 'Home':
	   st.subheader('Home')

	   st.write('''*....Welcome to our Clinical Decision Support System Project....*''')

	   

	elif choice == 'Login':
		st.subheader('Login Section')

		username = st.sidebar.text_input('username')
		password = st.sidebar.text_input('password',type='password')

		if st.sidebar.button('Login'):
			create_table()
			result = login_user(username,password)

			if result:

				st.success('Logged as {}'.format(username))

				task = st.selectbox('Task',['Prediction','Plot'])

				if task == 'Plot':
						st.subheader('Data Visualization Plot')

						df = pd.read_csv('projectDataset.csv')
						st.dataframe(df)
						
						if st.checkbox("Area Chart"):
							columns = df.columns.to_list()
							feature_choices = st.multiselect("Choose a Feature",columns)
							df_new = df[feature_choices]
							st.area_chart(df_new)



				elif task == 'Prediction':
					st.subheader('Prediction')

					st.subheader("Choose the Model")
				
					
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

		first_name = st.text_input('Firstname')
		surname = st.text_input('Surname')
		new_user = st.text_input('Username')
		email = st.text_input('Email')
		new_pasword = st.text_input('Password',type='password')
		confirm_pasword = st.text_input('Confirm Password ', type='password')

		if st.button('Signup'):
			create_table()
			add_userdata(first_name,surname,new_user,email,new_pasword)
			st.success('You have successfully created an account')
			st.info('Go to Login menu to login')


	elif choice == 'Manage Users':
		st.subheader('Admin Login')
		username = st.sidebar.text_input('username')
		password = st.sidebar.text_input('password',type='password')

		if st.sidebar.button('Login'):
				if password == 'admin123':
				   st.success("Welcome {}".format(username))
				
				   users_data = view_all_users()
				   users_db = pd.DataFrame(users_data,columns = ["username","password"])
				   st.dataframe(users_db)


				else:
					st.warning("Incorrect Username or Password")

if __name__ == '__main__':
	main()