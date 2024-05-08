import streamlit as st
import pandas as pd
from sklearn.svm import SVR
import base64


##Function to predict diabetes risk based on risk factor
def predict_diabetes_risk(risk_factor):
    # Assuming `svm_model` is defined and trained properly
    svm_model = SVR(kernel='rbf')
    # Load dataset into a dataframe
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')

    # Define features and target variable
    features = X_train[['num__Fruits', 'num__Non-starchy vegetables',
                        'num__Other starchy vegetables', 'num__Refined grains',
                        'num__Whole grains', 'num__Total processed meats',
                        'num__Unprocessed red meats', 'num__Eggs',
                        'num__Sugar-sweetened beverages', 'num__Fruit juices',
                        'num__Saturated fat', 'num__Monounsaturated fatty acids',
                        'num__Added sugars', 'num__Dietary sodium', 'num__Selenium',
                        'num__Total Milk', 'num__ObesityRate']]

    target = y_train['Diabetes prevalence (% of population ages 20 to 79)']
    # svm_model.fit(X_train, y_train)

    # Create a DataFrame for prediction
    data = pd.DataFrame({'Risk Factor': [risk_factor]})

    # Make prediction
    prediction = svm_model.predict(data)[0]  # Assuming predicting risk directly

    return prediction


##Function to display content based on page state

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def click_button():
    st.session_state.clicked = True


def display_page(page_num):
    if page_num == 1:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        st.markdown(
            '<p class="big-font">This application will be used to assess if you have any dietary risk factors for diabetes. It will ask you a series of questions regarding the types and amount of food you eat to provide a dietary risk assessment for type 2 diabetes. </p>',
            unsafe_allow_html=True)
        st.write("Getting treated for diabetes early can prevent the following life altering diseases:")
        st.image('complications.png')
        st.write(
            ":warning: The information provided by this application is not intended to be a substitute for professional medical advice, diagnosis, or treatment.")
        st.write(
            ":file_folder: The information entered in the application will not be stored or used for any other purposes.")
        name = st.text_input("Please enter your name:")
        st.session_state["name"] = name
        gender = st.radio('Gender Preference', ['Male', 'Female', 'Non-Binary'])

        Height = st.number_input("Please enter your height in cm", value=None, placeholder="Type a number eg 155...")
        st.session_state["Height"] = Height
        Weight = st.number_input("Please enter your weight in kg", value=None, placeholder="Type a number eg 65...")
        st.session_state["Weight"] = Weight

        submit_page1 = st.button("Submit", on_click=click_button)

        if submit_page1:
            ##st.session_state["name"] = name
            st.session_state["page"] = 2  # Move to page 2 after submission
            st.experimental_rerun()

    elif page_num == 2:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        st.write(f"Hi {st.session_state['name']}!")
        st.write("The early signs and symptoms of Type 2 Diabetes can include:")

        signs = [
            "Frequent urination :toilet:", "Increased thirst :potable_water:", "Fatigue	:sleeping:",
            "Blurred vision :eyeglasses:",
            "Slow healing of cuts and wounds :adhesive_bandage:", "Tingling numbness or pain in hands or feet :wave:",
            "Patches of darker skin :black_medium_square:", "Itching and yeast infections :mushroom:"
        ]

        st.markdown("\n".join([f"- {sign}" for sign in signs]))

        st.write(
            ":large_red_square: If you have recently started experiencing these symptoms, it is recommended that you seek medical advice as soon as possible.")
        st.link_button("To find a GP (General Practitioner) please click here",
                       "https://www.healthdirect.gov.au/australian-health-services")
        if st.button("NEXT"):
            st.session_state["page"] = 3
            st.experimental_rerun()

    elif page_num == 3:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        name = st.session_state["name"]
        st.write(f"{name}, what do you eat each day?")
        ##st.link_button("For more information on serving sizes please click here",
                       ##"https://www.eatforhealth.gov.au/food-essentials/five-food-groups/grain-cereal-foods-mostly-wholegrain-and-or-high-cereal-fibre")

        # Define the list of food options
        food_options = [
            "Eggs", "Fruit", "Non-starchy vegetable", "Starchy vegetable", "Refined grains",
            "Whole grains", "Processed meats", "Unprocessed meats", "Sweetened beverages",
            "Fruit juice", "Saturated fats", "Unsaturated fats", "Added sugars", "Added salts", "Dairy"
        ]





        st.write("How many serves of fruit per day?")
        st.image("fruitserve.png")
        serves_per_day_fruit = st.slider(f"How many serves of fruit per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_fruit"] = serves_per_day_fruit
        st.write("")

        st.write("How many serves of Non Starchy Vegetables per day?")
        st.image("nonstartchyserve.png")
        serves_per_day_nonstarchyveg = st.slider(f"How many serves of Non Starchy Vegetables per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_nonstarchyveg"] = serves_per_day_nonstarchyveg
        st.write("")



        st.write("How many serves of Starchy Vegetables per day?")
        st.image("starchyvegserve.png")
        serves_per_day_starchyveg = st.slider(f"How many serves of Starchy Vegetables per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_starchyveg"] = serves_per_day_starchyveg
        st.write("")

        st.write("How many serves of Dairy per day?")
        st.image("dairyserve.png")
        serves_per_day_dairy = st.slider(f"How many serves of Dairy per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_dairy"] = serves_per_day_dairy
        st.write("")

        st.write("How many serves of Refined Grains per day?")
        st.image("refgrainserve.png")
        serves_per_day_refgrain = st.slider(f"How many serves of Refined Grains per day?", min_value=0, max_value=6,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_refgrain"] = serves_per_day_refgrain
        st.write("")

        st.write("How many serves of Whole Grains per day?")
        st.image("whgrainserve.png")
        serves_per_day_whgrain = st.slider(f"How many serves of Whole Grains per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_whgrain"] = serves_per_day_whgrain
        st.write("")

        st.write("How many serves of Processed Meat per day?")
        st.image("prmeatserve.png")
        serves_per_day_prmeat = st.slider(f"How many serves of Processed Meat per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_prmeat"] = serves_per_day_prmeat
        st.write("")

        st.write("How many serves of Eggs per day?")
        st.image("eggserve.png")
        serves_per_day_egg = st.slider(f"How many serves of Eggs per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_egg"] = serves_per_day_egg
        st.write("")

        st.write("How many serves of Unprocessed Meat per day?")
        st.image("unprmeatserve.png")
        serves_per_day_unprmeat = st.slider(f"How many serves of Unprocessed Meat per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_unprmeat"] = serves_per_day_unprmeat
        st.write("")

        st.write("How many serves of Sweetened Beverages per day?")
        st.image("swdrinkserve.png")
        serves_per_day_swbeverage = st.slider(f"How many serves of Sweetened Beverages per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_swbeverage"] = serves_per_day_swbeverage
        st.write("")

        st.write("How many serves of Fruit Juice per day?")
        st.image("fjuiceserve.png")
        serves_per_day_fjuice = st.slider(f"How many serves of fruit juice per day?", min_value=0, max_value=4,  help=None, label_visibility="collapsed")
        st.session_state["serves_per_day_fjuice"] = serves_per_day_fjuice
        st.write("")
        # Add a button to submit selection
        if st.button("Submit"):
            # Calculate risk factor based on servings per day
            # result = calculate_risk_factor(serves_per_day)
            #
            # # Use a trained model to predict diabetes risk
            # risk_prediction = predict_diabetes_risk(result)
            #
            # st.session_state["results"] = risk_prediction
            # st.success("Selections submitted successfully")
            st.session_state["page"] = 4  # Move to results page
            st.experimental_rerun()

    elif page_num == 4:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns


        st.image('Header.png')

        data = pd.read_csv('GlobalDietaryDatabase_V2.csv')
        data = pd.concat([data] * 20)
        df_cleaned = data.copy()

        # Handle missing values. For simplicity, we'll fill missing values with the mean of their respective columns.
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

        # Check if there's any remaining missing value that wasn't handled (e.g., non-numeric columns).
        missing_values_check = df_cleaned.isnull().sum()

        # Check data types for a brief overview.
        data_types = df_cleaned.dtypes

        ##(df_cleaned.head(), missing_values_check, data_types)

        # Define dictionary of old and new column names
        column_name_mapping = {
            'Entity': 'Country',
            'Population (2021)': 'Population', 'Gross National Income Per Capita (2021)': 'Gross Income Per Capita'
        }

        # Rename the columns
        df_cleaned = df_cleaned.rename(columns=column_name_mapping)

        # split X and y into training and testing sets

        # Import necessary libraries
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # view dimensions of dataset
        df = df_cleaned


        from sklearn.model_selection import train_test_split

        X = df.drop(['Diabetes prevalence (% of population ages 20 to 79)'], axis=1)
        y = (df['Diabetes prevalence (% of population ages 20 to 79)'] > 7.0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        for col in X_train.columns:
            if X_train[col].dtype != 'object':
                missing_percentage = X_train[col].isnull().mean()
                if missing_percentage > 0:
                    print(f"{col}: {round(missing_percentage * 100, 2)}%")

        categorical_cols_to_drop = ['Country', 'superregion2', 'iso3']
        X_train.drop(categorical_cols_to_drop, axis=1, inplace=True)
        X_test.drop(categorical_cols_to_drop, axis=1, inplace=True)
        X.drop(categorical_cols_to_drop, axis=1, inplace=True)

        ##print(X_train.head)

        cols = X_train.columns

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        # Fit and transform on training data
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform the testing data
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrames with original column names
        X_train_scaled = pd.DataFrame(X_train, columns=cols)
        X_test_scaled = pd.DataFrame(X_test, columns=cols)

        X_train_scaled.head()

        from sklearn.svm import SVC

        # Create an instance of the SVM classifier
        svm_classifier = SVC()

        # Fit the classifier on your training data
        svm_classifier.fit(X_train_scaled, y_train)

        from sklearn.metrics import accuracy_score

        from sklearn.metrics import confusion_matrix
        # import pickle
        # model_pkl_file = "app/eatwelllivewell.pkl"
        #
        # with open(model_pkl_file, 'wb') as file:
        #     pickle.dump(svm_classifier, file)
        #
        # with open(model_pkl_file, 'rb') as file:
        #     model = pickle.load(file)

        model =svm_classifier

        serves_per_day_fruit= st.session_state["serves_per_day_fruit"]
        serves_per_day_starchyveg = st.session_state["serves_per_day_starchyveg"]
        serves_per_day_nonstarchyveg = st.session_state["serves_per_day_nonstarchyveg"]
        serves_per_day_dairy = st.session_state["serves_per_day_dairy"]
        serves_per_day_refgrain = st.session_state["serves_per_day_refgrain"]
        serves_per_day_whgrain = st.session_state["serves_per_day_whgrain"]
        serves_per_day_prmeat = st.session_state["serves_per_day_prmeat"]
        serves_per_day_unprmeat = st.session_state["serves_per_day_unprmeat"]
        serves_per_day_egg = st.session_state["serves_per_day_egg"]
        serves_per_day_swbeverage = st.session_state["serves_per_day_swbeverage"]
        serves_per_day_fjuice = st.session_state["serves_per_day_fjuice"]
        Height = st.session_state["Height"]
        Weight = st.session_state["Weight"]
        BMI = 51

        name = st.session_state["name"]

        fruit = float(serves_per_day_fruit) * 150
        starchyveg = float(serves_per_day_starchyveg) * 180
        ##nonstarchyveg = float(serves_per_day_nonstarchyveg) * 100
        dairy = float(serves_per_day_dairy) * 250
        refgrain = float(serves_per_day_refgrain) * 50
        whgrain = float(serves_per_day_whgrain) * 50
        prmeat = float(serves_per_day_prmeat) * 50
        unprmeat = float(serves_per_day_unprmeat) * 100
        egg = float(serves_per_day_egg) * 55
        swbeverage = float(serves_per_day_swbeverage) * 248
        fjuice = float(serves_per_day_fjuice) * 248
        BMI = float(Weight) / ((float(Height) / 100) ** 2)
        BMI = float(BMI)
        test = np.asarray([fruit,starchyveg,refgrain,whgrain,prmeat,unprmeat,egg,dairy,swbeverage,fjuice,BMI])
        test = test.reshape(1, -1)

        # ##with open(model_pkl_file, 'rb') as file:
        # import pickle
        # model = pickle.load(open(eatwelllivewell.pkl))
        # print(model)

        ## check unprocessed red meats
        inputmodel = pd.DataFrame(test, columns = ['Fruits', 'starchy vegetables', 'Refined grains', 'Whole grains',
        'Total processed meats', 'Unprocessed red meats', 'Eggs', 'Total Dairy',
        'Sugar-sweetened beverages', 'Fruit juices', 'BMI'])

        inputmodel.to_csv('inputmodel1.csv')

        y_predict = model.predict(inputmodel)
        ##st.write(f"{name}, what do you eat each day?")


        ##st.write(y_predict)

        if (y_predict ==1 ):
            st.write(
                f"Hi {name} based on the information you provided YOU ARE AT RISK for having diabetes. It is recommended that you seek medical advice")
            st.page_link("https://www.healthdirect.gov.au/australian-health-services",
                         label=":blue-background[To find a GP (General Practitioner) please click here]", icon="⚕️")
        else:
            st.write(f"Hi {name} you are NOT at risk for having diabetes.")

        ##st.write(y_predict)
        st.write('Based on the information you entered into this Application')
        def get_bmi_category(bmi):
            if bmi < 18.5:
                return "Underweight"
            elif bmi < 25:
                return "Normal weight"
            elif bmi < 30:
                return "Overweight"
            else:
                return "Obese"
        bmi_category = get_bmi_category(BMI)
        st.write(f' Your BMI is {BMI:.1f}  and this puts you in the category of {bmi_category}.')

        st.write("You are eating:")

        rec_serve_fruit = 2
        if serves_per_day_fruit + serves_per_day_fjuice < rec_serve_fruit:
            st.write(":arrow_down: LESS fruit (including juice) than the recommended 2 serves per day.")
        elif serves_per_day_fruit + serves_per_day_fjuice == rec_serve_fruit:
            st.write(":white_check_mark: The recommended serving size of 2 serves of fruit per day.")
        else:
            st.write(":arrow_up: MORE fruit than the recommended 2 serves per day.")

        rec_serve_vegies = 6
        if serves_per_day_starchyveg + serves_per_day_nonstarchyveg < rec_serve_vegies:
            st.write(":arrow_down: LESS vegetables than the recommended 6 serves per day.")
        elif serves_per_day_starchyveg + serves_per_day_nonstarchyveg == rec_serve_vegies:
            st.write(":white_check_mark: The recommended serving size of 6 serves of vegetables per day.")
        else:
            st.write(":arrow_up: MORE vegetables than the recommended 6 serves per day.")

        rec_serve_grain = 6
        total_grain = serves_per_day_refgrain + serves_per_day_whgrain

        if total_grain < rec_serve_grain:
            st.write(":arrow_down: LESS refined grains than the recommended 6 serves per day.")
        elif total_grain == rec_serve_grain:
            st.write(":white_check_mark: The recommended serving size of 6 serves of refined grains per day.")
        else:
            st.write(":arrow_up: MORE refined grains than the recommended 6 serves per day.")


        if total_grain > 0 and serves_per_day_whgrain / total_grain < 0.5:
            st.markdown(''':arrow_down: LESS whole grains than refined grains.  
                    :arrow_right: It is recommended that you increase your whole grains.''')

        elif total_grain > 0 and serves_per_day_whgrain / total_grain < 0.75:
            st.write(
                ":arrow_down: Between 50-75% whole grains and 25-50% refined grains. Try eating less refined grains.")
        else:
            st.write(
                ":white_check_mark: Mainly whole grains and less refined grains which exceeds the recommended average.")

        rec_serve_leanmeat = 2.5
        leanmeat = serves_per_day_unprmeat + serves_per_day_egg

        if leanmeat < rec_serve_leanmeat:
            st.write(":arrow_down: LESS unprocessed meat and eggs than the recommended 2.5 serves per day.")
        elif leanmeat == rec_serve_leanmeat:
            st.write(
                ":white_check_mark: The recommended serving size of 2.5 serves of unprocessed meat and eggs per day.")
        else:
            st.write(":arrow_up: MORE unprocessed meat and eggs than the recommended 2.5 serves per day.")

        rec_serve_dairy = 3

        if serves_per_day_dairy < rec_serve_dairy:
            st.write(":arrow_down: LESS dairy than the recommended 3 serves per day.")
        elif serves_per_day_dairy == rec_serve_dairy:
            st.write(":white_check_mark: The recommended serving size of 3 serves of dairy per day.")
        else:
            st.write(":arrow_up: MORE dairy than the recommended 3 serves per day.")

        rec_serve_discret = 2
        Total_discret = serves_per_day_prmeat + serves_per_day_swbeverage
        if Total_discret < rec_serve_discret:
            st.write(":arrow_down: LESS discretionary foods than the recommended 2 serves per day.")
        elif serves_per_day_dairy == rec_serve_dairy:
            st.write(":white_check_mark: The recommended serving size of 2 serves of discretionary foods per day.")
        else:
            st.markdown(''':arrow_up: MORE discretionary foods than the recommended 3 serves per day.  
                    :arrow_right:  It is recommended that your replace these servings with a more healthy option.''')
        st.markdown('''  :yum: Discretionary Foods are foods you enjoy but are unnecessary.  
                           :bubble_tea: For example sweetened beverages and processed meats''')

        st.markdown(f"Thanks {name} for using this application and I hope you found it useful.")
        st.page_link("https://www.diabetesaustralia.com.au/risk-calculator/",
                     label=":red-background[For more information about the risk factors of diabetes please click here]",
                     icon="ℹ️")

        st.markdown('<a href="mailto:Suellen.L.Fletcher@student.uts.edu.au">Feedback Welcome via Email </a>',
                    unsafe_allow_html=True)

        # 'Fruits', 'starchy vegetables', 'Refined grains', 'Whole grains',
        # 'Total processed meats', 'Unprocessed red meats', 'Eggs', 'Total Dairy',
        # 'Sugar-sweetened beverages', 'Fruit juices', 'BMI'],
        ##dtype = 'object'
                              ##,serves_per_day_starchyveg,serves_per_day_refgrain,serves_per_day_whgrain,serves_per_day_prmeat,serves_per_day_unprmeat,serves_per_day_egg,serves_per_day_dairy,serves_per_day_swbeverage,serves_per_day_fjuice,BMI])


        ##st.button("Restart1")
        submit_page4 = st.button("Restart the App", on_click=click_button)

        if submit_page4:
            ##st.session_state["name"] = name
            st.session_state["page"] = 1  # Move to page 2 after submission
            st.experimental_rerun()

# def calculate_risk_factor(serves_per_day):
#     risk_factor = 0
#     for food, serves in serves_per_day.items():
#         if serves:
#             try:
#                 serves = float(serves[:1])  # Extract numeric part from slider label
#                 if food in ["Refined grains", "Processed meats", "Sweetened beverages", "Added sugars", "Added salts"]:
#                     risk_factor += serves
#             except ValueError:
#                 st.warning(f"Invalid input for serves of {food}. Please enter a valid number.")
#     return risk_factor


# Set Streamlit page config
st.set_page_config(
    page_title="Eat Well Live Well",
    page_icon=":green_salad:",
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = 1

# Display content based on the current page
display_page(st.session_state["page"])

