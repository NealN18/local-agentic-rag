import re
import xml.etree.ElementTree as ET
import logging
from urllib.parse import urljoin, urlparse
from requests import Session
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def parse_llms_txt(text: str, base_url: str) -> list[str]:
    urls = []
    for match in re.finditer(r'\[.*?\]\((.*?)\)', text):
        url = match.group(1)
        if url.startswith("http") or url.startswith("/"):
            urls.append(urljoin(base_url, url))
    return list(set(urls))

def parse_sitemap(xml_text: str) -> list[str]:
    urls = []
    try:
        root = ET.fromstring(xml_text)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for loc in root.findall('.//ns:loc', ns):
            if loc.text: urls.append(loc.text.strip())
    except ET.ParseError:
        pass
    return urls

def fallback_html_spider(base_url: str, session: Session, max_pages: int = 20) -> list[str]:
    visited = set()
    queue = [base_url]
    discovered = []
    
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    
    while queue and len(discovered) < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        visited.add(url)
        
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue
                
            discovered.append(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            for a_tag in soup.find_all("a", href=True):
                next_url = urljoin(url, a_tag["href"]).split("#")[0] 
                parsed_next = urlparse(next_url)
                
                if parsed_next.netloc == base_domain and next_url not in visited and next_url not in queue:
                    queue.append(next_url)
        except Exception:
            pass
            
    return discovered

def discover_urls(base_url: str, session: Session, timeout: int) -> list[str]:
    base = base_url.rstrip('/')
    
    try:
        resp = session.get(f"{base}/llms.txt", timeout=timeout)
        if resp.status_code == 200 and "text/plain" in resp.headers.get("content-type", ""):
            urls = parse_llms_txt(resp.text, base_url)
            if urls:
                logger.info(f"Discovered {len(urls)} URLs via llms.txt")
                return urls
    except Exception: pass

    try:
        resp = session.get(f"{base}/sitemap.xml", timeout=timeout)
        if resp.status_code == 200 and "xml" in resp.headers.get("content-type", ""):
            urls = parse_sitemap(resp.text)
            if urls:
                logger.info(f"Discovered {len(urls)} URLs via sitemap.xml")
                return urls
    except Exception: pass

    logger.warning("No llms.txt or sitemap.xml found. Falling back to lightweight HTML spider.")
    urls = fallback_html_spider(base_url, session, max_pages=20)
    if urls:
        logger.info(f"Discovered {len(urls)} URLs via HTML spider")
        return urls
        
    logger.warning("No URLs found by spider. Returning base URL only.")
    return [base_url]