# LoanHub

**Team Members:**

Harivatsan Selvam

Krishanth Dineshkumar

Saad Agaria

Sonny Arangode

**Project Purpose:** Our project aims to streamline the loan application process using machine learning and a user-friendly interface. The system collects user details such as employment information, credit score, and debt-to-income ratio. It then submits loan requests to multiple banks simultaneously and predicts whether the user qualifies for a loan.

**Tools Utilized:**

Jupyter for Python-based machine learning development

Flask for frontend and UI integration

FlutterFlow for rapid app development and UI design

Visual Studio Code (VS Code) for development environment

FastAPI for creating efficient APIs

Firebase for database management

Google Colab for collaborative machine learning development

Scikit-learn for machine learning models


**Challenges Faced by the FlutterFlow App (Frontend/UI Team):** Signup Page Integration with Firebase: Initially, the signup page did not store email and password data in Firebase. We resolved this issue by implementing a backend query and ensuring proper authorization of email data through Firebase.

**Challenges faced by the backend team:** The AI model's datasets were highly imbalanced, requiring us to balance them using sampling techniques. To address this, we learned and applied methods such as SMOTE. The process was insightful, challenging, and enjoyable.

**Integration of ML Code from Jupyter to FlutterFlow App:** Transitioning the Python-based ML code from Jupyter to the FlutterFlow app posed a challenge due to database dependencies on Firebase. To address this, we developed a solution using FastAPI to create an API and connected it to Google Colab. This allowed us to import and integrate the ML code efficiently with FlutterFlow via FastAPI.

**Subsequent Challenge and Solution:**

**Efficient Input Mapping for ML Code:** Following the integration, we encountered difficulties in effectively obtaining user input from FlutterFlow and mapping it to the ML code. To overcome this hurdle, we opted for Flask as an alternative solution. Flask facilitated direct mapping of user inputs to the ML code, significantly enhancing system efficiency.

**Public Framework Used:** FastAPI was instrumental in creating efficient APIs for seamless integration between our FlutterFlow frontend and the machine learning models developed in Google Colab.

**Check out the LoanHub Video Demo!** https://drive.google.com/file/d/1cGjQD5ojQhtnKFyrhXcjwoGWHT1Ktb4V/view?usp=sharing


**Check out the FlutterFlow App Demo!** https://drive.google.com/file/d/1hJW9GseA6-bOHXXlh7r-rcUKkn5B-rAx/view?usp=sharing

