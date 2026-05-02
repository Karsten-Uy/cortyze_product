import instaloader

L = instaloader.Instaloader()
USER = "vebaxak634"

try:
    # This specifically looks for your logged-in session in Chrome
    L.interactive_login(USER) 
    L.save_session_to_file()
    print("✅ Session saved successfully!")
except Exception as e:
    print(f"❌ Failed: {e}")