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
        submit_page1 = st.button("Submit", on_click=click_button)

        if submit_page1:
            ##st.session_state["name"] = name
            st.session_state["page"] = 2  # Move to page 2 after submission

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

    elif page_num == 3:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        name = st.session_state["name"]
        st.write(f"{name}, what do you eat each day?")
        st.link_button("For more information on serving sizes please click here",
                       "https://www.eatforhealth.gov.au/food-essentials/five-food-groups/grain-cereal-foods-mostly-wholegrain-and-or-high-cereal-fibre")

        # Define the list of food options
        food_options = [
            "Eggs", "Fruit", "Non-starchy vegetable", "Starchy vegetable", "Refined grains",
            "Whole grains", "Processed meats", "Unprocessed meats", "Sweetened beverages",
            "Fruit juice", "Saturated fats", "Unsaturated fats", "Added sugars", "Added salts", "Dairy"
        ]

        BMI = st.slider(f"Enter BMI?",
                                  help="hep")
        st.session_state["BMI"] = BMI



        ##serves_per_day_fruit = {}

        st.image("fruitserve.png", caption="Serving Size Fruit")
        serves_per_day_fruit = st.select_slider(f"How many serves of fruit per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        ##serves_per_day_fruit = float(serves_per_day_fruit[:1]) * 150

        st.session_state["serves_per_day_fruit"] = serves_per_day_fruit

        st.image("starchyvegserve.png", caption="Serving Size Starchy Vegetables")
        serves_per_day_starchyveg = st.select_slider(f"How many serves of starchy vegetables per day?",
                                  options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_starchyveg"] = serves_per_day_starchyveg



        st.image("dairyserve.png", caption="Serving Size Dairy")
        serves_per_day_dairy = st.select_slider(f"How many serves of dairy per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_dairy"] = serves_per_day_dairy

        st.image("refgrainserve.png", caption="Serving Size Refined Grains")
        serves_per_day_refgrain = st.select_slider(f"How many serves of refined grains per day?",
                                  options=["0-1", "2-3", "4-5", "6 or more"],
                                  help="hep")
        st.session_state["serves_per_day_refgrain"] = serves_per_day_refgrain


        st.image("whgrainserve.png", caption="Serving Size Whole Grains")
        serves_per_day_whgrain = st.select_slider(f"How many serves of whole grains per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_whgrain"] = serves_per_day_whgrain


        st.image("prmeatserve.png", caption="Serving Size Processed Meats")
        serves_per_day_prmeat = st.select_slider(f"How many serves of processed meat per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_prmeat"] = serves_per_day_prmeat


        st.image("eggserve.png", caption="Serving Size Eggs")
        serves_per_day_egg = st.select_slider(f"How many serves of eggs per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_egg"] = serves_per_day_egg


        st.image("unprmeatserve.png", caption="Serving Size Unprocessed Meat")
        serves_per_day_unprmeat = st.select_slider(f"How many serves of Unprocessed Meat per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_unprmeat"] = serves_per_day_unprmeat


        st.image("swdrinkserve.png", caption="Serving Size Sweetened Beverage")
        serves_per_day_swbeverage = st.select_slider(f"How many serves of Sweetened Beverage per day?",
                                  options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_swbeverage"] = serves_per_day_swbeverage


        st.image("fjuiceserve.png", caption="Serving Size Fruit Juice")
        serves_per_day_fjuice = st.select_slider(f"How many serves of Fruit Juice per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        st.session_state["serves_per_day_fjuice"] = serves_per_day_fjuice


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
        serves_per_day_dairy = st.session_state["serves_per_day_dairy"]
        serves_per_day_refgrain = st.session_state["serves_per_day_refgrain"]
        serves_per_day_whgrain = st.session_state["serves_per_day_whgrain"]
        serves_per_day_prmeat = st.session_state["serves_per_day_prmeat"]
        serves_per_day_unprmeat = st.session_state["serves_per_day_unprmeat"]
        serves_per_day_egg = st.session_state["serves_per_day_egg"]
        serves_per_day_swbeverage = st.session_state["serves_per_day_swbeverage"]
        serves_per_day_fjuice = st.session_state["serves_per_day_fjuice"]
        BMI = st.session_state["BMI"]

        name = st.session_state["name"]

        fruit = float(serves_per_day_fruit[:1]) * 150
        starchyveg = float(serves_per_day_starchyveg[:1]) * 180
        dairy = float(serves_per_day_dairy[:1]) * 250
        refgrain = float(serves_per_day_refgrain[:1]) * 50
        whgrain = float(serves_per_day_whgrain[:1]) * 50
        prmeat = float(serves_per_day_prmeat[:1]) * 50
        unprmeat = float(serves_per_day_unprmeat[:1]) * 100
        egg = float(serves_per_day_egg[:1]) * 55
        swbeverage = float(serves_per_day_swbeverage[:1]) * 248
        fjuice = float(serves_per_day_fjuice[:1]) * 248
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

        y_predict = model.predict(inputmodel)
        ##st.write(f"{name}, what do you eat each day?")


        ##st.write(y_predict)

        if (y_predict ==1 ):
            st.write(f"Hi {name} based on the information you provided YOU ARE AT RISK for having diabetes. It is recommended that you seek medical advice")
            st.link_button("To find a GP (General Practitioner) please click here",
                           "https://www.healthdirect.gov.au/australian-health-services")

        else:
            st.write(f"Hi {name} you are NOT at risk for having diabetes.")










        # 'Fruits', 'starchy vegetables', 'Refined grains', 'Whole grains',
        # 'Total processed meats', 'Unprocessed red meats', 'Eggs', 'Total Dairy',
        # 'Sugar-sweetened beverages', 'Fruit juices', 'BMI'],
        ##dtype = 'object'
                              ##,serves_per_day_starchyveg,serves_per_day_refgrain,serves_per_day_whgrain,serves_per_day_prmeat,serves_per_day_unprmeat,serves_per_day_egg,serves_per_day_dairy,serves_per_day_swbeverage,serves_per_day_fjuice,BMI])


        ##st.button("Restart1")
        submit_page4 = st.button("Submit", on_click=click_button)

        if submit_page4:
            ##st.session_state["name"] = name
            st.session_state["page"] = 1  # Move to page 2 after submission

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

