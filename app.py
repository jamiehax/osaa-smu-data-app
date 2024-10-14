import streamlit as st
from helper_functions import setup_db
import hmac

# check password
def check_password():
    """
    Returns `True` if the user had the correct password.
    """

    def password_entered():
        """
        Checks whether a password entered by the user is correct.
        """

        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # remove the password
        else:
            st.session_state["password_correct"] = False

    # return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # show input for password.
    st.image("content/OSAA-Data-logo.svg")

    st.markdown("Welcome to the Office of the Speical Advisor to Africa's Strategic Management Unit's Data App. Please enter the app password to access the data app.")

    st.text_input(
        "Password",
        placeholder="enter the app password...",
        on_change=password_entered,
        key="password",
        label_visibility="collapsed"
    )

    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()


# create test database
db_path = setup_db()

if 'db_path' not in st.session_state:
    st.session_state.db_path = db_path

# create app pages
home_page = st.Page("home.py", title="Home", icon=":material/home:")
dashboard_page = st.Page("dashboard.py", title="Data Dashboard", icon=":material/analytics:")
wb_dashboard_page = st.Page("wb_dashboard.py", title="WorldBank Data Dashboard", icon=":material/bar_chart:")
sdg_dashboard_page = st.Page("sdg_dashboard.py", title="SDG Data Dashboard", icon=":material/show_chart:")
chatbot_page = st.Page("chatbot.py", title="OSAA General Chatbot", icon=":material/chat:")
contradictory_analysis_page = st.Page("check_analysis.py", title="Contradictory Analysis Tool", icon=":material/check:")
pid_checker_page = st.Page("pid_checker.py", title="PID Checker", icon=":material/task_alt:")

pg = st.navigation([home_page, dashboard_page, wb_dashboard_page, sdg_dashboard_page, chatbot_page, contradictory_analysis_page, pid_checker_page])
st.set_page_config(page_title="SMU Data App", page_icon=":material/home:", layout="wide")
pg.run()