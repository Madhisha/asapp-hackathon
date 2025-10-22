from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import os

# --- Helper Functions ---

def clean_text(text):
    """Utility to clean whitespace from extracted text."""
    if not text:
        return ""
    cleaned = text.replace('\xa0', ' ').strip()
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned_lines = [line for line in lines if line]
    # Join lines back, handling list items slightly better
    final_text = ""
    for i, line in enumerate(cleaned_lines):
        final_text += line
        if i < len(cleaned_lines) - 1:
            # Add newline if it looks like a list item, otherwise space
            if line.startswith(('-', '•', '')) or cleaned_lines[i+1].startswith(('-', '•', '')):
                     final_text += "\n"
            else:
                     final_text += " "
    return ' '.join(final_text.split()) # Final cleanup

def clean_answer_text_preserve_newlines(text):
    """Cleans up whitespace while preserving intentional newlines for lists."""
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = [line for line in lines if line]
    return '\n'.join(cleaned_lines)

# --- Fares Page Parsing ---

def parse_fare_table(soup_section, context_name):
    """
    Parses the fare comparison table and converts it into a
    list of Question-Answer pairs for the RAG, using the
    provided context (e.g., tab name).
    """
    print(f"         - Parsing fare table for context: '{context_name}'")
    qa_pairs = []
    table_div = soup_section.select_one('jb-table div[role="table"].dn.db-ns')
    if not table_div:
        # Try finding any visible table as a fallback
        table_div = soup_section.select_one('jb-table div[role="table"]')
    
    if not table_div:
        print(f"         - Info: Could not find the main fare comparison table structure for '{context_name}'.")
        return None

    headers = []
    header_row = table_div.select_one('div[role="rowgroup"] > div[role="row"]')
    if header_row:
        header_cells = header_row.select('div[role="columnheader"]')
        # Get fare types (e.g., "Blue Basic", "Blue")
        headers = [clean_text(th.get_text(strip=True)) for th in header_cells[1:]] 

    if not headers:
        print(f"         - Warning: Could not extract table headers for '{context_name}'.")
        return None

    body_row_group = table_div.select_one('div[role="rowgroup"]:nth-of-type(2)')
    if not body_row_group:
        print(f"         - Warning: Could not find table body row group for '{context_name}'.")
        return None

    rows = body_row_group.select('div[role="row"]')
    
    for row in rows:
        cells = row.select('div[role="cell"]')
        if not cells: continue

        # Get the feature name (e.g., "Checked bag(s) included")
        feature_name_tag = cells[0].find('div', class_='s-body')
        feature_name = clean_text(feature_name_tag.get_text(strip=True)) if feature_name_tag else clean_text(cells[0].get_text(strip=True))
        
        if not feature_name: continue

        # Iterate through each fare type for this feature
        for i, header in enumerate(headers):
            if (i + 1) < len(cells):
                cell_content = cells[i + 1].get_text(separator=' ', strip=True)
                answer = clean_text(cell_content)
                
                # Create a specific Q&A pair with context
                if feature_name and header and (answer or answer == "0"): # Allow "0" as a valid answer
                    question = f"For a '{context_name}', what is the policy for '{feature_name}' with a '{header}' fare?"
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
            
    return qa_pairs if qa_pairs else None


def parse_faqs(soup_section):
    """Parses Question (h3) and Answer (following p/ul) pairs."""
    faq_list = []
    # Look within specific containers first
    qa_containers = soup_section.select('jb-body-text-container jb-inner-html.lh-copy')
    if not qa_containers:
        # Fallback to broader search if specific structure not found
        qa_containers = soup_section.select('jb-body-text jb-inner-html.lh-copy')
        
    print(f"         - Info: Found {len(qa_containers)} potential Q&A containers for FAQs.")

    processed_faq_answers = set() # Avoid duplicates within FAQs

    for container in qa_containers:
        question_tag = container.find('h3')
        if question_tag:
            question = clean_text(question_tag.get_text(strip=True))
            answer_elements = question_tag.find_next_siblings(['p', 'ul'])
            answer_parts = []
            for elem in answer_elements:
                if elem.name == 'h3': break
                answer_parts.append(elem.get_text(separator='\n', strip=True))

            if answer_parts:
                answer = clean_answer_text_preserve_newlines('\n'.join(answer_parts))
                if question and answer and answer not in processed_faq_answers:
                    faq_list.append({"question": question, "answer": answer})
                    processed_faq_answers.add(answer)

    print(f"         - Info: Extracted {len(faq_list)} unique FAQ pairs.")
    return faq_list if faq_list else None

# --- Pet Travel Page Parsing (MODIFIED) ---

def parse_pet_tab_panel(panel_soup):
    """
    (NEW FUNCTION)
    Parses the content of a single pet travel tab panel.
    """
    answer_tag = panel_soup.find('jb-inner-html') # Content is inside this
    if answer_tag:
        answer = clean_answer_text_preserve_newlines(answer_tag.get_text(strip=True, separator='\n'))
        return answer
    
    # Fallback: just get all text from the panel if jb-inner-html isn't found
    answer = clean_answer_text_preserve_newlines(panel_soup.get_text(strip=True, separator='\n'))
    return answer if answer else None

def parse_pet_page_static_content(soup_section):
    """
    (RENAMED & MODIFIED)
    Parses non-tabbed Q&A (Checklist, general FAQs) for the Pet Travel page.
    Returns a list of {"question": ..., "answer": ...} dicts.
    """
    qa_pairs = []
    processed_answers = set()

    # --- Strategy 1: Specific Sections (like Checklist) ---
    try:
        checklist_section = soup_section.find('div', id='pet-travel-checklist')
        if checklist_section:
            question_tag = checklist_section.find('h2')
            intro_p = question_tag.find_next_sibling('p') if question_tag else None
            answer_tag = checklist_section.find('jb-inner-html') # List is inside this

            if question_tag and answer_tag:
                question = clean_text(question_tag.get_text(strip=True))
                intro_text = clean_answer_text_preserve_newlines(intro_p.get_text(strip=True, separator='\n')) if intro_p else ""
                list_text = clean_answer_text_preserve_newlines(answer_tag.get_text(strip=True, separator='\n'))
                
                # Combine intro and list for the answer
                answer = f"{intro_text}\n{list_text}".strip()

                if question and answer and answer not in processed_answers:
                    qa_pairs.append({"question": question, "answer": answer})
                    processed_answers.add(answer)
                    print(f"         - Info: Parsed 'Pet Travel Checklist' section.")
    except Exception as e:
        print(f"         - Warning: Could not parse specific 'Pet Travel Checklist' section. {e}")

    # --- Strategy 2: Tabbed Content (REMOVED) ---
    # This is now handled by the main scrape_policy_pages function using Selenium.


    # --- Strategy 3: General Q&A Blocks (h3 -> p/ul) ---
    qa_containers = soup_section.select('jb-body-text-container jb-inner-html.lh-copy, jb-body-text jb-inner-html.lh-copy')
    print(f"         - Info: Found {len(qa_containers)} potential general Q&A containers.")
    
    initial_qa_count = len(qa_pairs)

    for container in qa_containers:
        question_tag = container.find('h3')
        if question_tag:
            question = clean_text(question_tag.get_text(strip=True))
            
            answer_elements = question_tag.find_next_siblings(['p', 'ul'])
            answer_parts = []
            for elem in answer_elements:
                if elem.name == 'h3': break # Stop at next question
                answer_parts.append(elem.get_text(separator='\n', strip=True))

            if answer_parts:
                answer = clean_answer_text_preserve_newlines('\n'.join(answer_parts))
                # Add if question, answer exist and answer is unique
                if question and answer and answer not in processed_answers:
                    qa_pairs.append({"question": question, "answer": answer})
                    processed_answers.add(answer)
                        
    found_general_qa = len(qa_pairs) - initial_qa_count
    print(f"         - Info: Extracted {found_general_qa} unique general Q&A pairs.")
    
    return qa_pairs if qa_pairs else None


# --- Main Scraping Function ---

def scrape_policy_pages(urls, wait_time=12):
    """
    Scrape policy pages using Selenium, parsing tables & FAQs for 'Fares',
    Q&A for 'Pet Travel', and text for others.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=chrome_options)
    policies = {}

    for section, url in urls.items():
        print(f"\nScraping section '{section}' from {url} ...")
        try:
            driver.get(url)
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
            time.sleep(3) # Extra wait for initial JS rendering
        except Exception as e:
            print(f"   - Error: Failed to load page or find main content for '{section}'. Skipping. Error: {e}")
            policies[section] = f"Error: Could not load page or find main content. {e}"
            continue

        section_data = None

        # --- Fares Page Logic (No changes) ---
        if section == "Fares":
            print(f"   - Parsing specific content for '{section}'...")
            section_data_list = [] # Initialize an empty list
            
            # 1. Find all fare tabs
            try:
                tab_buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'][id^='jb-tab-id-']")
                if not tab_buttons:
                    print("     - Info: No tabs found. Parsing as a single page.")
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    main_content = soup.find('main')
                    
                    table_result = parse_fare_table(main_content, "General") # Pass a default context
                    if table_result:
                        section_data_list.extend(table_result)
                        print(f"     - Success: Parsed table into {len(table_result)} Q&A pairs.")
                    else:
                        print(f"     - Warning: Could not parse fare table.")
                else:
                    print(f"     - Info: Found {len(tab_buttons)} tabs. Iterating...")
                    # 2. Iterate through each tab, click, and parse
                    for i in range(len(tab_buttons)):
                        tab_buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'][id^='jb-tab-id-']")
                        tab_button = tab_buttons[i]
                        
                        try:
                            tab_name = clean_text(tab_button.text)
                            if not tab_name:
                                tab_name = f"Tab {i+1}" # Fallback
                                
                            print(f"     - Clicking tab: '{tab_name}'")
                            
                            driver.execute_script("arguments[0].click();", tab_button)
                            time.sleep(3) 
                            
                            soup = BeautifulSoup(driver.page_source, 'html.parser')
                            main_content = soup.find('main')
                            
                            if not main_content:
                                print(f"     - Error: Could not find main content after clicking tab '{tab_name}'.")
                                continue

                            table_result = parse_fare_table(main_content, tab_name)
                            if table_result:
                                section_data_list.extend(table_result)
                                print(f"         - Success: Parsed table for '{tab_name}', got {len(table_result)} Q&A pairs.")
                            else:
                                print(f"         - Warning: Could not parse fare table for '{tab_name}'.")
                                
                        except Exception as e:
                            tab_name_for_error = tab_name if 'tab_name' in locals() else f"Tab {i}"
                            print(f"     - Error processing '{tab_name_for_error}': {e}")
                            continue # Skip to next tab

            except Exception as e:
                print(f"     - Error finding tabs: {e}. Will attempt to parse as a single page.")
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                main_content = soup.find('main')
                table_result = parse_fare_table(main_content, "General") # Pass a default context
                if table_result:
                    section_data_list.extend(table_result)
                    print(f"     - Success: Parsed table into {len(table_result)} Q&A pairs.")

            # 4. Parse the regular FAQs (which are likely outside the tabs)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            main_content = soup.find('main')
            if main_content:
                faq_result = parse_faqs(main_content)
                if faq_result:
                    section_data_list.extend(faq_result)
                    print(f"     - Success: Parsed {len(faq_result)} general FAQs.")
                else:
                    print(f"     - Info: No general FAQs found or failed to parse.")
            else:
                 print("     - Warning: Could not find main content to parse FAQs.")

            # 5. Check if we gathered any Q&A pairs at all
            if not section_data_list:
                print(f"     - Warning: No structured data found for '{section}'. Falling back to general text.")
                if 'main_content' not in locals() or not main_content:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    main_content = soup.find('main')
                
                if main_content:
                    section_data = clean_text(main_content.get_text(separator=' ', strip=True)) # Fallback to text
                    print(f"     - Scraped {len(section_data)} characters (approx) as fallback.")
                else:
                    section_data = "Error: Fallback failed, main content not found."
            else:
                section_data = section_data_list # Assign the combined list

        # --- Pet Travel Page Logic (HEAVILY MODIFIED) ---
        elif section == "Pet Travel":
            print(f"   - Parsing specific content for '{section}'...")
            section_data_list = []
            
            # 1. Parse static content (Checklist, general FAQs) first
            try:
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                main_content = soup.find('main')
                if main_content:
                    # Use the modified function that only gets static content
                    static_qa = parse_pet_page_static_content(main_content) 
                    if static_qa:
                        section_data_list.extend(static_qa)
                        print(f"     - Success: Parsed {len(static_qa)} static Q&A pairs (Checklist, FAQs).")
                else:
                    print("     - Warning: Could not find main content for static parsing.")
            except Exception as e:
                print(f"     - Warning: Error parsing static content: {e}")

            # 2. Find and parse dynamic tabs
            try:
                tab_buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'][id^='jb-tab-id-']")
                if not tab_buttons:
                    print("     - Info: No dynamic tabs found.")
                else:
                    print(f"     - Info: Found {len(tab_buttons)} tabs. Iterating...")
                    processed_tab_answers = set() # Avoid duplicates from tabs
                    
                    for i in range(len(tab_buttons)):
                        # Re-find elements to avoid stale reference
                        tab_buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='tab'][id^='jb-tab-id-']")
                        tab_button = tab_buttons[i]
                        
                        try:
                            tab_name = clean_text(tab_button.text) # This is the QUESTION
                            tab_panel_id = tab_button.get_attribute('aria-controls')
                            
                            if not tab_name or not tab_panel_id:
                                print(f"     - Warning: Skipping tab {i} (no name or panel ID).")
                                continue

                            print(f"     - Clicking tab: '{tab_name}'")
                            driver.execute_script("arguments[0].click();", tab_button)
                            
                            # Wait for the panel content to be loaded/visible
                            time.sleep(2) 

                            # Find the specific panel by its ID
                            panel_element = driver.find_element(By.ID, tab_panel_id)
                            panel_html = panel_element.get_attribute('innerHTML')
                            panel_soup = BeautifulSoup(panel_html, 'html.parser')
                            
                            # Use the new helper function to parse the panel's content
                            answer = parse_pet_tab_panel(panel_soup) 
                            
                            if tab_name and answer and answer not in processed_tab_answers:
                                section_data_list.append({"question": tab_name, "answer": answer})
                                processed_tab_answers.add(answer)
                                print(f"         - Success: Parsed Q&A for tab '{tab_name}'.")
                            elif not answer:
                                print(f"         - Info: Tab '{tab_name}' had no content.")
                                
                        except Exception as e:
                            tab_name_for_error = tab_name if 'tab_name' in locals() else f"Tab {i}"
                            print(f"     - Error processing '{tab_name_for_error}': {e}")
                            continue # Skip to next tab
            
            except Exception as e:
                print(f"     - Error finding tabs: {e}.")

            # 3. Final check and fallback
            if not section_data_list:
                print(f"   - Warning: No structured data found for '{section}'. Falling back to general text.")
                # Find main_content again in case it was lost in error paths
                if 'main_content' not in locals() or not main_content:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    main_content = soup.find('main')
                
                if main_content:
                    section_data = clean_text(main_content.get_text(separator=' ', strip=True)) # Fallback to text
                    print(f"     - Scraped {len(section_data)} characters (approx) as fallback.")
                else:
                    section_data = "Error: Fallback failed, main content not found."
            else:
                section_data = section_data_list # Assign the combined list

        
        # --- Default text extraction for other pages (No changes) ---
        else:
            print(f"   - Extracting general text content for '{section}'...")
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            main_content = soup.find('main')
            if not main_content:
                 print(f"   - Error: Could not find main content tag for '{section}'. Skipping.")
                 policies[section] = "Error: Main content tag not found."
                 continue
                 
            # Try specific content blocks first
            content_blocks = main_content.find_all(['jb-body-text', 'jb-inner-html'])
            text = ""
            if content_blocks:
                processed_texts = []
                for block in content_blocks:
                    block_text = block.get_text(separator='\n', strip=True)
                    lines = [line.strip() for line in block_text.split('\n')]
                    cleaned_lines = [line for line in lines if line]
                    processed_texts.append('\n'.join(cleaned_lines))
                text = '\n\n'.join(processed_texts)
                
            if not text.strip():
                print(f"     - Info: No specific content blocks found. Using main text fallback for '{section}'.")
                text = clean_text(main_content.get_text(separator=' ', strip=True))

            section_data = text
            print(f"     - Scraped {len(text)} characters (approx) for '{section}'.")

        policies[section] = section_data

    driver.quit()
    print("\nFinished scraping all sections.")
    return policies

# --- Saving Function (No changes needed) ---
def save_policies(policies, filename='policies.json'):
    """Saves the scraped policies dictionary to a JSON file."""
    filepath = os.path.join(os.getcwd(), filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(policies, f, indent=2, ensure_ascii=False)
        print(f"Policies saved to {filepath}")
    except IOError as e:
        print(f"Error saving policies to {filepath}: {e}")
    except TypeError as e:
        print(f"Error during JSON serialization: {e}")

# --- Main Execution Block (No changes needed) ---
if __name__ == "__main__":
    urls = {
        "Fares": "https://www.jetblue.com/flying-with-us/our-fares",
        "Pet Travel": "https://www.jetblue.com/traveling-together/traveling-with-pets"
    }

    policies = scrape_policy_pages(urls)
    save_policies(policies)