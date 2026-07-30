from http.client import PROXY_AUTHENTICATION_REQUIRED
from http.client import ResponseNotReady
from requests import request
from http.client import responses
import requests  #Yeh sabse popular Python library hai HTTP requests bhejne ke liye.Isko real-world APIs (Jaise Twitter, Weather, ChatGPT) call karne ke liye use karte hain.
import json# Python ka built-in module. API se response JSON string mein aata hai, json.loads() use karte hain. But requests ka .json() method khud parse kar deta hai, toh hum json module ko mostly printing/formatting ke liye use kar rahe hain.
from bs4 import BeautifulSoup
from config import Config  # Humari .env file se URL load karega.
from logger import setup_logger #Humare khud ke logger function ko import karte hain. 

# Initialize logger for this module
# __name__ ensures logs show 'data_ingestion' as the source
logger = setup_logger(__name__,log_file='logs/pipeline.log')


def fetch_data_from_api():
    url = Config.BASE_URL

    logger.info(f"🌐 Attempting API call: {url}")

    try:
        # ⭐ STAR: requests.get() is the standard way to make HTTP GET requests.
        # timeout=5 ensures we don't wait forever if the server is down.
        response = requests.get(url,timeout=5) # Production me timeout mandatory hai.
        

        # ⭐ STAR: raise_for_status() checks HTTP status code.
        # If status is 404 or 500, it raises an HTTPError exception automatically.
        # This saves us from writing manual 'if status != 200' checks.
        response.raise_for_status()

        # ⭐ STAR: .json() parses the JSON response into Python dict/list
        data = response.json()

        logger.info(f"")
        return data

    except requests.exceptions.Timeout:
        logger.error("❌ API request timed out after 5 seconds.")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP Error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        # Catch-all for any other requests library errors (network, SSL, etc.)
        logger.error(f"❌ Network/Request Error: {e}")
        return None
    except Exception as e:
        # Catch any unexpected Python errors (e.g., JSON decode error)
        logger.critical(f"❌ Unexpected error in API call: {e}")
        return None

        

def fetch_data_from_web_scrper():

    logger.warning("⚠️ API failed. Switching to Web Scraping fallback...")
    url = "http://quotes.toscrape.com/"  # A safe, static site for practice

    try:
        responses = requests.get(url,timeout=10)
        responses.raise_for_status()
        # HTML text ko parse karke tree structure banata hai. 'html.parser' Python ka built-in parser hai.
        #  (Isme kuch install nahi karna padta).
        soup = BeautifulSoup(responses.text,'html.parser')

        #find_all(): ⭐ STAR. HTML tags extract karne ka sabse common method. Syntax: tag_name,
        #  class_ (underscore note karo kyunki class Python keyword hai).
        quotes = soup.find_all('span',class_='text')
        #Ye ek list return karta hai. Agar kuch nahi milta toh empty list aati hai (error nahi aata).
        authors = soup.find_all('small',class_='author')

        scraped_data = []

        for q,a in zip(quotes,authors):
            scraped_data.append(
                {"quote":q.text.strip() ,
                "author": a.text.strip()}
            )
        

        logger.info(f"✅ Scraping Success: Fetched {len(scraped_data)} quotes.")
        return scraped_data

        #q.text: Tag ke andar ka text nikalta hai.
        #.strip(): Extra spaces hata deta hai (cleaning).
        #zip(quotes, authors): Quotes aur authors ko pair karta hai. 
        # Agar 2 quotes aur 3 authors hain, toh sirf 2 pairs banenge (safely ignore).

    except requests.exceptions.RequestException as e:
        # Catch-all for any other requests library errors (network, SSL, etc.)
        logger.error(f"❌ Network/Request Error: {e}")
        return None
    except Exception as e:
        # Catch any unexpected Python errors (e.g., JSON decode error)
        logger.critical(f"❌ Unexpected error in API call: {e}")
        return None


if __name__ == "__main__":
    data = fetch_data_from_api()
    if data is None:
        data = fetch_data_from_web_scrper()
    
    if data:
        print(f"✅ Final Data: {data}")
        for i,record in enumerate(data,start=1):
            print(f"{i}.{record}")        
    else:
        print("❌ Failed to fetch data from all sources.")