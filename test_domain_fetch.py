import re
import logging
def extract_domain(url):
    """Extract the primary domain name from a URL using regex."""
    try:
        # Regex to match domain (e.g., 'amazon' from 'https://www.amazon.com')
        pattern = r'^(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)'
        match = re.match(pattern, url)
        if not match:
            raise ValueError(f"Cannot extract domain from URL: {url}")
        domain = match.group(1).split('.')[-2]  # Get primary domain (e.g., 'amazon' from 'amazon.com')
        logging.info(f"Extracted domain: {domain}")
        print(domain)
        return domain
    except Exception as e:
        logging.error(f"Failed to extract domain from URL {url}: {e}")
        raise

if __name__ =="__main__":
    extract_domain("https://www.target.com/s?searchTerm=paper+plates&category=0%7CAll%7Cmatchallpartial%7Call+categories&searchTermRaw=paper+")
    extract_domain("https://www.walmart.com/search?q=lenovo+laptop")

    

## PASS WORKS FOR US TO ADD IN THE MANAGE.PY