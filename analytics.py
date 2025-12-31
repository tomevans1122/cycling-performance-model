import streamlit as st

def inject_ga():
    GA_ID = "G-YXB3FZ1SLL"

    # Note: We use st.markdown with unsafe_allow_html=True.
    # This executes the JS in the main browser window, not an iframe.
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        // Debug message to verify script execution
        console.log("Initializing Google Analytics for {GA_ID}...");

        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        gtag('config', '{GA_ID}');
        
        console.log("Google Analytics Loaded!");
    </script>
    """
    
    # Inject into the app
    st.markdown(ga_code, unsafe_allow_html=True)
