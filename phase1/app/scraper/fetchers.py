import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.robotparser import RobotFileParser
from app.scraper.policies import DomainPolicy

logger = logging.getLogger(__name__)

def build_session(policy: DomainPolicy) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Phase4-Universal-RAGBot/1.0"})
    return session

def check_robots(url: str, policy: DomainPolicy, session: requests.Session) -> bool:
    if not policy.respect_robots: return True
    try:
        rp = RobotFileParser()
        rp.set_url(f"{url.scheme}://{url.netloc}/robots.txt")
        resp = session.get(rp.url, timeout=policy.timeout)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            return rp.can_fetch(session.headers["User-Agent"], url.geturl())
    except Exception: pass
    return True

def fetch_static(url: str, policy: DomainPolicy, session: requests.Session) -> str | None:
    try:
        resp = session.get(url, timeout=policy.timeout)
        resp.raise_for_status()
        time.sleep(1.0 / policy.rate_limit_rps) # Polite rate limiting
        return resp.text
    except Exception as e:
        logger.warning(f"Static fetch failed for {url}: {e}")
        return None

def fetch_playwright(url: str, policy: DomainPolicy) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=policy.timeout * 1000)
            time.sleep(1.0) 
            
            captcha_selectors = [
                "iframe[src*='recaptcha']", "iframe[src*='hcaptcha']",
                "#cf-challenge-form", "#challenge-form",
                ".g-recaptcha", ".h-captcha"
            ]
            for sel in captcha_selectors:
                if page.query_selector(sel):
                    logger.warning(f"CAPTCHA detected on {url}. Aborting fetch to remain compliant.")
                    return None
                    
            title = page.title().lower()
            if any(kw in title for kw in ["captcha", "just a moment", "attention required"]):
                logger.warning(f"Challenge/CAPTCHA title detected on {url}. Aborting.")
                return None
                
            html = page.content()
            time.sleep(1.0 / policy.rate_limit_rps)
            return html
        finally:
            browser.close()