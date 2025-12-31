import streamlit as st
import streamlit.components.v1 as components

def inject_ga():
    """
    Injects GA4 code using a frontend component.
    This method is compatible with read-only filesystems (like Streamlit Cloud).
    """
    
    # This HTML includes the standard GA4 tag provided by Google
    ga_js = """
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-YXB3FZ1SLL"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-YXB3FZ1SLL');
    </script>
    """
    
    # Inject the HTML invisibly into the page
    # height=0 ensures it doesn't take up visual space
    components.html(ga_js, height=0, width=0)
