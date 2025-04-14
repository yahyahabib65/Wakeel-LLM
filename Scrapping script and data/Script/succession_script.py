from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

# Setup Chrome
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Go to login page
driver.get("https://www.pakistanlawsite.com/Login/MainPage")

# Wait for login fields to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.form-control")))

# Fill in username and password
inputs = driver.find_elements(By.CSS_SELECTOR, "input.form-control")
inputs[0].send_keys("AGTLLEGAL")  # Username
inputs[1].send_keys("659867")     # Password

# Agree to terms
driver.find_element(By.CSS_SELECTOR, "input[type='checkbox']").click()

# Click Sign in
driver.find_element(By.CSS_SELECTOR, "button.btn.btn-success").click()

# Wait for page to load
time.sleep(5)

# Click the "Latest Statutes" JavaScript-triggered link
try:
    latest_statutes = driver.find_element(By.LINK_TEXT, "Latest Statutes")
    latest_statutes.click()
    print("✅ Clicked 'Latest Statutes'")
except Exception as e:
    print("❌ Could not click 'Latest Statutes':", e)
    driver.quit()
    exit()

# Wait for letter index and click "S"
try:
    WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.LINK_TEXT, "S"))).click()
    print("✅ Clicked on 'S'")
except Exception as e:
    print("❌ Failed to click 'S':", e)
    driver.quit()
    exit()

# Wait for and click "Succession Act 1925"
try:
    WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Succession Act 1925"))).click()
    print("✅ Clicked 'Succession Act 1925'")
except Exception as e:
    print("❌ Could not click 'Succession Act 1925':", e)
    driver.quit()
    exit()

# Wait for list of sections to load
WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.LINK_TEXT, "Read")))

# Get all "Read" buttons
read_buttons = driver.find_elements(By.LINK_TEXT, "Read")

# Scroll to each and extract the pop-up text
for i in range(len(read_buttons)):
    read_buttons = driver.find_elements(By.LINK_TEXT, "Read")
    section_row = read_buttons[i].find_element(By.XPATH, "./ancestor::tr")
    section_title = section_row.find_elements(By.TAG_NAME, "td")[1].text.strip().replace(" ", "_")

    # Click "Read" to open modal
    read_buttons[i].click()

    # Wait for modal
    modal = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "modal-content"))
    )

    # Extract text
    content = modal.text.strip()

    # Save to file
    filename = f"{section_title}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"💾 Saved: {filename}")

    # Close modal
    driver.find_element(By.CLASS_NAME, "close").click()

    # Brief pause
    time.sleep(1)

# Done
print("✅ All sections saved.")
driver.quit()
