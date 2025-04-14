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
options.add_argument("--start-maximized")  # Optional: start in fullscreen
driver = webdriver.Chrome(options=options)

# Go to login page
driver.get("https://www.pakistanlawsite.com/Login/MainPage")

# Wait for page to load
WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "username")))

# Fill in login form
driver.find_element(By.NAME, "username").send_keys("AGTLLEGAL")
driver.find_element(By.NAME, "password").send_keys("659867")

# Accept terms and conditions checkbox (usually named "accept" or similar)
driver.find_element(By.NAME, "accept").click()

# Submit login form
driver.find_element(By.NAME, "submit").click()

# Wait for home page
WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.LINK_TEXT, "Statutes Search")))

# Click on Statutes Search
driver.find_element(By.LINK_TEXT, "Statutes Search").click()

# Wait for search index to load and click "S"
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.LINK_TEXT, "S"))).click()

# Click "Succession Act 1925"
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Succession Act 1925"))).click()

# Wait for list of sections to load
WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.LINK_TEXT, "Read")))

# Get all "Read" buttons
read_buttons = driver.find_elements(By.LINK_TEXT, "Read")

# Scroll to each and extract the pop-up text
for i in range(len(read_buttons)):
    # Re-find elements every loop (to avoid stale reference)
    read_buttons = driver.find_elements(By.LINK_TEXT, "Read")
    section_row = read_buttons[i].find_element(By.XPATH, "./ancestor::tr")
    section_title = section_row.find_elements(By.TAG_NAME, "td")[1].text.strip().replace(" ", "_")

    # Click "Read" to open modal
    read_buttons[i].click()

    # Wait for modal to appear
    modal = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "modal-content"))
    )

    # Extract text from modal
    content = modal.text.strip()

    # Save to file
    filename = f"{section_title}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved: {filename}")

    # Close modal
    driver.find_element(By.CLASS_NAME, "close").click()

    # Short wait before next click
    time.sleep(1)

# Done
print("✅ All sections saved.")
driver.quit()
