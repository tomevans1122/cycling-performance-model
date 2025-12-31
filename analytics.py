import streamlit as st
import streamlit.components.v1 as components

def inject_ga():
    GA_ID = "G-YXB3FZ1SLL"

    # This JavaScript does three things:
    # 1. It breaks out of the Streamlit iframe (using window.parent).
    # 2. It checks if the tag is already there (to prevent duplicates).
    # 3. It manually creates the <script> tag and forces the browser to download it.
    
    gtag_js = f"""
    <script>
        // Use the parent window (the main app)
        var parentDoc = window.parent.document;
        
        // Check if we already injected the tag to avoid double-counting
        if (!parentDoc.getElementById('ga-injected')) {{
            
            console.log("Injecting GA4 for {GA_ID} into main page...");

            // 1. Create the library script
            var script = parentDoc.createElement('script');
            script.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
            script.async = true;
            script.id = 'ga-injected'; // Mark as injected
            
            // 2. Append it to the HEAD of the main page
            parentDoc.head.appendChild(script);

            // 3. Initialize the Data Layer on the main window
            window.parent.dataLayer = window.parent.dataLayer || [];
            function gtag(){{ window.parent.dataLayer.push(arguments); }}
            
            gtag('js', new Date());
            gtag('config', '{GA_ID}');
            
            console.log("GA4 Injection Complete. Network request should follow.");
        }}
    </script>
    """
    
    # Render this JS invisibly
    components.html(gtag_js, height=0, width=0)
