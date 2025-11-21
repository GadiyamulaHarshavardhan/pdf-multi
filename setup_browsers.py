"""
Setup script for browser automation tools
"""

import subprocess
import sys
import os

def setup_selenium():
    """Install and setup Selenium WebDriver"""
    print("🔧 Setting up Selenium...")
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium import webdriver
        
        # Install ChromeDriver
        driver_path = ChromeDriverManager().install()
        print(f"✅ ChromeDriver installed at: {driver_path}")
        
        # Test Selenium
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        print("✅ Selenium test successful")
        driver.quit()
        
    except Exception as e:
        print(f"❌ Selenium setup failed: {e}")

def setup_playwright():
    """Install Playwright browsers"""
    print("🔧 Setting up Playwright...")
    try:
        import playwright
        # Install browsers
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "firefox"], check=True)
        print("✅ Playwright browsers installed")
        
    except Exception as e:
        print(f"❌ Playwright setup failed: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up browser automation tools...")
    
    setup_selenium()
    setup_playwright()
    
    print("✅ Setup completed!")

if __name__ == "__main__":
    main()