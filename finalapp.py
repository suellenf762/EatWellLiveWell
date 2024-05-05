import streamlit as st
import pandas as pd
from sklearn.svm import SVR
import base64

##Function to predict diabetes risk based on risk factor
def predict_diabetes_risk(risk_factor):
    # Assuming `svm_model` is defined and trained properly
    svm_model = SVR(kernel='rbf')
    #Load dataset into a dataframe
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
    background-image: url("data:image/png;base64,%s");
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
        st.write(":warning: The information provided by this application is not intended to be a substitute for professional medical advice, diagnosis, or treatment.")
        st.write(":file_folder: The information entered in the application will not be stored or used for any other purposes.")
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
            "Frequent urination :toilet:", "Increased thirst :potable_water:", "Fatigue	:sleeping:", "Blurred vision :eyeglasses:",
            "Slow healing of cuts and wounds :adhesive_bandage:", "Tingling numbness or pain in hands or feet :wave:",
            "Patches of darker skin :black_medium_square:", "Itching and yeast infections :mushroom:"
        ]

        st.markdown("\n".join([f"- {sign}" for sign in signs]))

        st.write(":large_red_square: If you have recently started experiencing these symptoms, it is recommended that you seek medical advice as soon as possible.")
        st.link_button("To find a GP (General Practitioner) please click here", "https://www.healthdirect.gov.au/australian-health-services")
        if st.button("NEXT"):
            st.session_state["page"] = 3

    elif page_num == 3:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        name = st.session_state["name"]
        st.write(f"{name}, what do you eat each day?")
        st.link_button("For more information on serving sizes please click here", "https://www.eatforhealth.gov.au/food-essentials/five-food-groups/grain-cereal-foods-mostly-wholegrain-and-or-high-cereal-fibre")
        

        # Define the list of food options
        food_options = [
            "Eggs", "Fruit", "Non-starchy vegetable", "Starchy vegetable", "Refined grains",
            "Whole grains", "Processed meats", "Unprocessed meats", "Sweetened beverages",
            "Fruit juice", "Saturated fats", "Unsaturated fats", "Added sugars", "Added salts", "Dairy"
        ]
        serves_per_day_fruit = {}

        st.image("fruitserve.png", caption="Serving Size Fruit")
        serves = st.select_slider(f"How many serves of fruit per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_fruit = float(serves[:1])*150
        serves_per_day_fruit

        st.image("starchyvegserve.png", caption="Serving Size Starchy Vegetables")
        serves = st.select_slider(f"How many serves of starchy vegetables per day?",
                                  options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_starchyveg = float(serves[:1])*180
        serves_per_day_starchyveg

        st.image("dairyserve.png", caption="Serving Size Dairy")
        serves = st.select_slider(f"How many serves of dairy per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_dairy = float(serves[:1])*250
        serves_per_day_dairy

        st.image("refgrainserve.png", caption="Serving Size Refined Grains")
        serves = st.select_slider(f"How many serves of refined grains per day?", options=["0-1", "2-3","4-5", "6 or more"],
                                  help="hep")
        serves_per_day_refgrain = float(serves[:1])*50
        serves_per_day_refgrain

        st.image("whgrainserve.png", caption="Serving Size Whole Grains")
        serves = st.select_slider(f"How many serves of whole grains per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_whgrain = float(serves[:1])*50
        serves_per_day_whgrain

        st.image("prmeatserve.png", caption="Serving Size Processed Meats")
        serves = st.select_slider(f"How many serves of processed meat per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_prmeat = float(serves[:1])*50
        serves_per_day_prmeat

        st.image("eggserve.png", caption="Serving Size Eggs")
        serves = st.select_slider(f"How many serves of eggs per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_egg = float(serves[:1])*55
        serves_per_day_egg

        st.image("unprmeatserve.png", caption="Serving Size Unprocessed Meat")
        serves = st.select_slider(f"How many serves of Unprocessed Meat per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_unprmeat = float(serves[:1])*100
        serves_per_day_unprmeat

        st.image("swdrinkserve.png", caption="Serving Size Sweetened Beverage")
        serves = st.select_slider(f"How many serves of Sweetened Beverage per day?",
                                  options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_swbeverage = float(serves[:1])*248
        serves_per_day_swbeverage

        st.image("fjuiceserve.png", caption="Serving Size Fruit Juice")
        serves = st.select_slider(f"How many serves of Fruit Juice per day?", options=["0-1", "2-3", "4 or more"],
                                  help="hep")
        serves_per_day_fjuice = float(serves[:1])*248
        serves_per_day_fjuice


        # Add a button to submit selection
        if st.button("Submit"):
            # Calculate risk factor based on servings per day
            result = calculate_risk_factor(serves_per_day)
            
            # Use a trained model to predict diabetes risk
            risk_prediction = predict_diabetes_risk(result)
            
            st.session_state["results"] = risk_prediction
            st.success("Selections submitted successfully")
            st.session_state["page"] = 4  # Move to results page

    elif page_num == 4:
        ##set_background('background.png')
        st.image('Header.png')
        ##st.title("Eat Well Live Well")
        st.write("Your predicted risk of developing Type 2 Diabetes:")
        st.write(st.session_state["results"])
        st.button("Restart")
        if st.button("Restart"):
            st.session_state["page"] = 1  # Restart to page 1

def calculate_risk_factor(serves_per_day):
    risk_factor = 0
    for food, serves in serves_per_day.items():
        if serves:
            try:
                serves = float(serves[:1])  # Extract numeric part from slider label
                if food in ["Refined grains", "Processed meats", "Sweetened beverages", "Added sugars", "Added salts"]:
                    risk_factor += serves
            except ValueError:
                st.warning(f"Invalid input for serves of {food}. Please enter a valid number.")
    return risk_factor

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
