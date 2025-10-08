#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import json
import hashlib
import logging
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque

from bs4 import BeautifulSoup, NavigableString
from playwright.sync_api import sync_playwright, Page
from khmercut import tokenize

# -------- Configuration --------
CONFIG = {
    "output_file": "khmer_dataset.jsonl",
    "min_articles": 10000,
    "min_words": 150,
    "candidate_limit": 50000,
    "request_timeout": 15,
    "sleep_interval": 0.5,
    "user_agent": "Mozilla/5.0 (compatible; KhmerScraper/1.0; +http://example.com/bot)",
    "seed_sites": [
        "https://plus.freshnewsasia.com/",
    ],
    "article_link_patterns": ["/article/", "/news/", "/post/", "/detail/", re.compile(r'/\d{4,}/\d{2,}/')],
    "pagination_tokens": ["page", "p=", "/page/"],
    "anon_patterns": [
        (re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.UNICODE), "[email protected]"),
        (re.compile(r'(\+?855[\d\-\s]{6,12}|0\d{7,9}|០[០-៩]{8,9})', re.UNICODE), "[phone]"),
        (re.compile(r'\b\d{5,}\b', re.UNICODE), "[number]"),
        (re.compile(r'(?<!\[img_)https?://[^\s,]+', re.UNICODE), "[url]"),
    ],
    "remove_selectors": [
        'script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe',
        '[class*="share"]', '[id*="share"]', '[class*="social"]', '[id*="social"]',
        '[class*="related"]', '[id*="related"]', '[class*="comment"]', '[id*="comment"]',
        '[class*="advert"]', '[id*="advert"]', '[class*="cookie"]', '[id*="cookie"]',
        '[class*="promo"]', '[class*="subscribe"]', '[class*="sidebar"]', '[id*="sidebar"]'
    ]
}

# -------- Logging Setup --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------- Helper Functions --------
def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def normalize_url(u: str) -> str:
    try:
        parts = urlparse(u)
        return urlunparse(parts._replace(query="", fragment=""))
    except Exception:
        return u

def fetch_html_with_playwright(page, url: str) -> str:
    try:
        page.goto(url, timeout=30000, wait_until='domcontentloaded')
        page.wait_for_timeout(1000)
        return page.content()
    except Exception as e:
        logging.warning("Playwright navigation failed for %s: %s", url, e)
        return ""

def is_article_link(url: str) -> bool:
    lower_url = url.lower()
    if any(path in lower_url for path in ['/tag/', '/category/', '/author/', '/ajax/']):
        return False
    if any(ext in lower_url for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.mp4', '.mp3']):
        return False
    for pattern in CONFIG["article_link_patterns"]:
        if isinstance(pattern, str) and pattern in lower_url:
            return True
        elif hasattr(pattern, 'search') and pattern.search(lower_url):
            return True
    path = urlparse(url).path
    if path and (path.count('/') >= 2) and re.search(r'\d{6,}', path):
         return True
    return False

def clean_and_prepare_soup(soup: BeautifulSoup):
    for selector in CONFIG["remove_selectors"]:
        for element in soup.select(selector):
            element.decompose()

def convert_elements_to_text(content_element: BeautifulSoup, base_url: str):
    for img in content_element.find_all('img'):
        src = img.get("data-src") or img.get("src")

        if not src:
            img.decompose()
            continue

        lower_src = src.lower()
        if any(keyword in lower_src for keyword in ['/ad/', 'banner', 'logo', 'icon', 'pixel', '.gif', '.svg', 'advertise']):
            logging.debug("Ignoring ad/icon image: %s", src)
            img.decompose()
            continue

        try:
            width = int(img.get('width', '100'))
            height = int(img.get('height', '100'))
            if width < 50 or height < 50:
                logging.debug("Ignoring small image: %s", src)
                img.decompose()
                continue
        except (ValueError, TypeError):
            pass

        full_url = urljoin(base_url, src)
        placeholder = f"[img_{full_url}]"
        img.replace_with(NavigableString(placeholder))

    for table in content_element.find_all('table'):
        table_text = []
        for row in table.find_all('tr'):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(['th', 'td'])]
            table_text.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(table_text) + "\n"))

def find_best_content_block(soup: BeautifulSoup) -> BeautifulSoup:
    candidates = []
    for element in soup.find_all(['div', 'article', 'main', 'section']):
        text = element.get_text(" ", strip=True)
        if not text or len(text) < 50:
            continue
        p_tags = len(element.find_all('p', recursive=False))
        text_length = len(text)
        link_density = len(element.find_all('a')) / (text_length + 1)
        score = text_length * (1 + p_tags * 0.5) * (1 - link_density)
        class_id_str = ' '.join(element.get('class', [])) + ' ' + element.get('id', '')
        if any(keyword in class_id_str for keyword in ['article-content', 'content-detail', 'article-description']):
            score *= 2.0
        elif any(keyword in class_id_str for keyword in ['content', 'article', 'post', 'entry', 'body']):
            score *= 1.5
        if any(keyword in class_id_str for keyword in ['comment', 'menu', 'share', 'nav', 'sidebar', 'footer', 'header']):
            score *= 0.5
        candidates.append((score, element))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def process_text(text: str) -> str:
    img_placeholders = re.findall(r'(\[img_.*?\])', text)
    text_no_imgs = re.sub(r'\[img_.*?\]', '@@IMAGE@@', text)

    temp_text = text_no_imgs

    # Remove leading timestamp like "25-09-2025 09:22"
    timestamp_pattern = re.compile(r'^\s*\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s*')
    temp_text = timestamp_pattern.sub('', temp_text)
    
    # --- NEW: Fix incorrect newline after dateline like (ភ្នំពេញ)៖ ---
    dateline_pattern = re.compile(r'^(\s*\([\u1780-\u17FF\s]+\)៖)\s*\n+')
    temp_text = dateline_pattern.sub(r'\1 ', temp_text)
    # ---

    for pattern, replacement in CONFIG["anon_patterns"]:
        temp_text = pattern.sub(replacement, temp_text)
    
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    temp_text = emoji_pattern.sub('', temp_text)
    temp_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', temp_text)
    temp_text = re.sub(r'[ \t]+', ' ', temp_text)
    temp_text = re.sub(r'\n{3,}', '\n\n', temp_text)

    final_text = temp_text
    if img_placeholders:
        placeholder_iter = iter(img_placeholders)
        final_text = re.sub('@@IMAGE@@', lambda m: next(placeholder_iter), temp_text)
    
    return final_text.strip()

def is_valid_content(text: str) -> bool:
    segmented_words = tokenize(text)
    word_count = len(segmented_words)

    if word_count < CONFIG["min_words"]:
        logging.warning("Article too short (%d words).", word_count)
        return False

    sample = text[:1000]
    khmer_chars = len(re.findall(r'[\u1780-\u17FF]', sample))
    total_chars = len(re.findall(r'\S', sample))
    if total_chars == 0 or (khmer_chars / total_chars) < 0.5:
        logging.warning("Content does not appear to be predominantly Khmer.")
        return False
    return True

def extract_article(html: str, url: str, domain: str = "NEWS"):
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    title = ""
    for selector in ["h1", ".entry-title", ".post-title", ".article-title", "title"]:
        tag = soup.select_one(selector)
        if tag:
            title_text = tag.get_text(" ", strip=True)
            if title_text:
                title = title_text
                break
    
    if not title:
        logging.warning("Could not extract title for %s", url)
        return None

    clean_and_prepare_soup(soup)
    content_element = find_best_content_block(soup)
    
    if not content_element:
        logging.warning("Could not find a suitable content block for %s", url)
        return None
        
    for heading in content_element.find_all(['h1', 'h2', 'h3', 'h4']):
        # Use `in` for partial matches, as titles might have extra whitespace
        if title in heading.get_text(" ", strip=True):
            heading.decompose()
            
    convert_elements_to_text(content_element, url)
    text = content_element.get_text("\n", strip=True)
    final_text = process_text(text)
    if not is_valid_content(final_text):
        logging.warning("Validation failed for %s", url)
        return None
    return {
        "id": md5_hex(url),
        "title": process_text(title),
        "text": final_text,
        "domain": domain,
        "url": url,
    }

def crawl_website(seed_url: str, page: Page, max_articles: int, existing_hashes: set):
    queue = deque([seed_url])
    visited = {seed_url}
    processed_articles = []

    domain = urlparse(seed_url).netloc

    while queue and len(processed_articles) < max_articles:
        current_url = queue.popleft()
        logging.info("Crawling: %s", current_url)
        html = fetch_html_with_playwright(page, current_url)
        if not html:
            continue

        soup = BeautifulSoup(html, "lxml")

        for a in soup.find_all("a", href=True):
            href = a['href']
            full_url = normalize_url(urljoin(current_url, href))

            if urlparse(full_url).netloc != domain or full_url in visited:
                continue

            visited.add(full_url)

            if is_article_link(full_url):
                logging.info("Found potential article: %s", full_url)
                article_html = fetch_html_with_playwright(page, full_url)
                article_data = extract_article(article_html, full_url, "NEWS")

                if article_data:
                    content_hash = md5_hex(article_data['text'])
                    if content_hash not in existing_hashes:
                        processed_articles.append(article_data)
                        existing_hashes.add(content_hash)
                        logging.info("Successfully extracted article #%d from %s", len(processed_articles), full_url)
                        if len(processed_articles) >= max_articles: break
                    else:
                        logging.info("Skipping duplicate content from %s", full_url)
            elif full_url not in queue:
                logging.debug("Adding to crawl queue: %s", full_url)
                queue.append(full_url)

        time.sleep(CONFIG["sleep_interval"])

    return processed_articles

def main():
    parser = argparse.ArgumentParser(description="Khmer Web Scraper")
    parser.add_argument('--url', type=str, help='Single article URL to scrape.')
    parser.add_argument('--seed', type=str, help='Seed URL to start a crawl (e.g., "https://news.sabay.com.kh/").')
    parser.add_argument('--output', type=str, default=CONFIG["output_file"], help='Output JSONL file path.')
    parser.add_argument('--max', type=int, default=CONFIG["min_articles"], help='Target number of articles for a crawl.')
    args = parser.parse_args()
    out_path = Path(args.output).expanduser().resolve()
    logging.info("Output will be saved to: %s", out_path)
    existing_hashes = set()
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_hashes.add(md5_hex(data.get("text", "")))
                except json.JSONDecodeError:
                    continue
    logging.info("Loaded %d existing content hashes.", len(existing_hashes))
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=CONFIG["user_agent"])
        page = context.new_page()
        try:
            with open(out_path, 'a', encoding='utf-8') as f:
                if args.url:
                    html = fetch_html_with_playwright(page, args.url)
                    article = extract_article(html, args.url)
                    if article:
                        f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        logging.info("Successfully processed and saved single URL.")
                    else:
                        logging.error("Failed to process the provided URL.")
                elif args.seed:
                    articles = crawl_website(args.seed, page, args.max, existing_hashes)
                    for article in articles:
                        f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    logging.info("Crawl complete. Saved %d new articles.", len(articles))
                else:
                    total_written = 0
                    for seed in CONFIG["seed_sites"]:
                        if total_written >= args.max: break
                        logging.info("--- Starting crawl for seed: %s ---", seed)
                        needed = args.max - total_written
                        articles = crawl_website(seed, page, needed, existing_hashes)
                        for article in articles:
                            f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        total_written += len(articles)
                        logging.info("Finished crawl for %s. Wrote %d articles. Total: %d", seed, len(articles), total_written)
        finally:
            browser.close()
    logging.info("Scraping finished.")

if __name__ == "__main__":
    main()