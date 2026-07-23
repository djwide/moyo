"""Press releases crawler for gathering company announcements and press releases."""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode, urljoin
import json
import re
import asyncio
import httpx

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

from moyo.publicside.gatherpublicsources.schema import PressRelease, PublicSource, SourceType, SearchQuery
from shared_utils import generate_id

logger = logging.getLogger(__name__)


class PressReleasesCrawler:
    """Crawler for press releases and company announcements."""
    
    def __init__(self):
        """Initialize the press releases crawler."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Press release sources
        self.sources = {
            'prnewswire': {
                'base_url': 'https://www.prnewswire.com',
                'search_url': 'https://www.prnewswire.com/search/news/',
                'api_url': 'https://www.prnewswire.com/api/search'
            },
            'businesswire': {
                'base_url': 'https://www.businesswire.com',
                'search_url': 'https://www.businesswire.com/portal/site/home/search/',
                'api_url': 'https://www.businesswire.com/api/search'
            },
            'globenewswire': {
                'base_url': 'https://www.globenewswire.com',
                'search_url': 'https://www.globenewswire.com/search/',
                'api_url': 'https://www.globenewswire.com/api/search'
            },
            'yahoo_finance': {
                'base_url': 'https://finance.yahoo.com',
                'search_url': 'https://finance.yahoo.com/news/',
                'api_url': 'https://query2.finance.yahoo.com/v8/finance/search'
            }
        }
    
    def search_press_releases(self, query: SearchQuery) -> List[PressRelease]:
        """Search for press releases using the given query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of press releases
        """
        press_releases = []
        
        try:
            # Search different sources
            sources_to_search = query.filters.get('sources', ['prnewswire', 'businesswire'])
            
            for source in sources_to_search:
                if source in self.sources:
                    releases = self._search_source(source, query)
                    press_releases.extend(releases)
            
            logger.info(f"Found {len(press_releases)} press releases for query: {query.query}")
            
        except Exception as e:
            logger.error(f"Error searching press releases: {e}")
        
        return press_releases[:query.max_results]
    
    def _search_source(self, source: str, query: SearchQuery) -> List[PressRelease]:
        """Search a specific press release source."""
        try:
            if source == 'prnewswire':
                return self._search_prnewswire(query)
            elif source == 'businesswire':
                return self._search_businesswire(query)
            elif source == 'globenewswire':
                return self._search_globenewswire(query)
            elif source == 'yahoo_finance':
                return self._search_yahoo_finance(query)
            else:
                logger.warning(f"Unsupported source: {source}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching {source}: {e}")
            return []
    
    def _search_prnewswire(self, query: SearchQuery) -> List[PressRelease]:
        """Search PR Newswire."""
        press_releases = []
        
        try:
            # PR Newswire search parameters
            search_params = {
                'keyword': query.query,
                'page': 1,
                'pageSize': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['startDate'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['endDate'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Add company filters
            if 'companies' in query.filters:
                companies = query.filters['companies']
                if companies:
                    search_params['company'] = companies[0]  # PR Newswire supports one company at a time
            
            # Make search request
            search_url = f"{self.sources['prnewswire']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            press_releases.extend(self._parse_prnewswire_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching PR Newswire: {e}")
        
        return press_releases
    
    def _search_businesswire(self, query: SearchQuery) -> List[PressRelease]:
        """Search Business Wire."""
        press_releases = []
        
        try:
            # Business Wire search parameters
            search_params = {
                'search': query.query,
                'page': 1,
                'limit': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['start_date'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['end_date'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Make search request
            search_url = f"{self.sources['businesswire']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            press_releases.extend(self._parse_businesswire_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching Business Wire: {e}")
        
        return press_releases
    
    def _search_globenewswire(self, query: SearchQuery) -> List[PressRelease]:
        """Search Globe Newswire."""
        press_releases = []
        
        try:
            # Globe Newswire search parameters
            search_params = {
                'q': query.query,
                'page': 1,
                'size': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['from'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['to'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Make search request
            search_url = f"{self.sources['globenewswire']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            press_releases.extend(self._parse_globenewswire_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching Globe Newswire: {e}")
        
        return press_releases
    
    def _search_yahoo_finance(self, query: SearchQuery) -> List[PressRelease]:
        """Search Yahoo Finance news."""
        press_releases = []
        
        try:
            # Yahoo Finance search parameters
            search_params = {
                'q': query.query,
                'type': 'news',
                'count': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['start'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['end'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Make search request
            search_url = f"{self.sources['yahoo_finance']['api_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            press_releases.extend(self._parse_yahoo_finance_results(data))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching Yahoo Finance: {e}")
        
        return press_releases
    
    def _parse_prnewswire_results(self, soup: BeautifulSoup) -> List[PressRelease]:
        """Parse PR Newswire search results."""
        press_releases = []
        
        try:
            # Look for press release articles
            articles = soup.find_all('article', class_='news-release')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.find('h3', class_='news-release-title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract company
                    company_elem = article.find('span', class_='company-name')
                    company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
                    
                    # Extract date
                    date_elem = article.find('time')
                    release_date = datetime.now()
                    if date_elem:
                        date_str = date_elem.get('datetime', '')
                        if date_str:
                            release_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    # Extract content preview
                    content_elem = article.find('p', class_='news-release-summary')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract URL
                    link_elem = article.find('a', href=True)
                    url = urljoin(self.sources['prnewswire']['base_url'], link_elem['href']) if link_elem else None
                    
                    press_release = PressRelease(
                        title=title,
                        company=company,
                        release_date=release_date,
                        content=content,
                        industry=None,
                        tags=[]
                    )
                    
                    press_releases.append(press_release)
                    
                except Exception as e:
                    logger.error(f"Error parsing PR Newswire article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing PR Newswire results: {e}")
        
        return press_releases
    
    def _parse_businesswire_results(self, soup: BeautifulSoup) -> List[PressRelease]:
        """Parse Business Wire search results."""
        press_releases = []
        
        try:
            # Look for press release articles
            articles = soup.find_all('div', class_='news-item')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.find('h3', class_='news-title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract company
                    company_elem = article.find('span', class_='company')
                    company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
                    
                    # Extract date
                    date_elem = article.find('span', class_='date')
                    release_date = datetime.now()
                    if date_elem:
                        date_str = date_elem.get_text(strip=True)
                        # Parse date string (format may vary)
                        try:
                            release_date = datetime.strptime(date_str, '%B %d, %Y')
                        except:
                            pass
                    
                    # Extract content preview
                    content_elem = article.find('p', class_='summary')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract URL
                    link_elem = article.find('a', href=True)
                    url = urljoin(self.sources['businesswire']['base_url'], link_elem['href']) if link_elem else None
                    
                    press_release = PressRelease(
                        title=title,
                        company=company,
                        release_date=release_date,
                        content=content,
                        industry=None,
                        tags=[]
                    )
                    
                    press_releases.append(press_release)
                    
                except Exception as e:
                    logger.error(f"Error parsing Business Wire article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Business Wire results: {e}")
        
        return press_releases
    
    def _parse_globenewswire_results(self, soup: BeautifulSoup) -> List[PressRelease]:
        """Parse Globe Newswire search results."""
        press_releases = []
        
        try:
            # Look for press release articles
            articles = soup.find_all('div', class_='news-item')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.find('h2', class_='title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract company
                    company_elem = article.find('span', class_='company')
                    company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
                    
                    # Extract date
                    date_elem = article.find('time')
                    release_date = datetime.now()
                    if date_elem:
                        date_str = date_elem.get('datetime', '')
                        if date_str:
                            release_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    # Extract content preview
                    content_elem = article.find('p', class_='excerpt')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    press_release = PressRelease(
                        title=title,
                        company=company,
                        release_date=release_date,
                        content=content,
                        industry=None,
                        tags=[]
                    )
                    
                    press_releases.append(press_release)
                    
                except Exception as e:
                    logger.error(f"Error parsing Globe Newswire article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Globe Newswire results: {e}")
        
        return press_releases
    
    def _parse_yahoo_finance_results(self, data: Dict[str, Any]) -> List[PressRelease]:
        """Parse Yahoo Finance search results."""
        press_releases = []
        
        try:
            if 'news' in data:
                for news_item in data['news']:
                    try:
                        title = news_item.get('title', 'No title')
                        company = news_item.get('publisher', 'Unknown company')
                        
                        # Extract date
                        release_date = datetime.now()
                        if 'published' in news_item:
                            release_date = datetime.fromtimestamp(news_item['published'])
                        
                        content = news_item.get('summary', '')
                        
                        press_release = PressRelease(
                            title=title,
                            company=company,
                            release_date=release_date,
                            content=content,
                            industry=None,
                            tags=[]
                        )
                        
                        press_releases.append(press_release)
                        
                    except Exception as e:
                        logger.error(f"Error parsing Yahoo Finance news item: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance results: {e}")
        
        return press_releases
    
    def get_press_release_details(self, url: str) -> Optional[PressRelease]:
        """Get detailed information for a specific press release.
        
        Args:
            url: URL of the press release
            
        Returns:
            Detailed press release information
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract detailed information based on the source
            if 'prnewswire.com' in url:
                return self._parse_prnewswire_detail(soup, url)
            elif 'businesswire.com' in url:
                return self._parse_businesswire_detail(soup, url)
            elif 'globenewswire.com' in url:
                return self._parse_globenewswire_detail(soup, url)
            else:
                return self._parse_generic_detail(soup, url)
                
        except Exception as e:
            logger.error(f"Error getting press release details: {e}")
            return None
    
    def _parse_prnewswire_detail(self, soup: BeautifulSoup, url: str) -> Optional[PressRelease]:
        """Parse detailed PR Newswire press release."""
        try:
            # Extract title
            title_elem = soup.find('h1', class_='news-release-title')
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract company
            company_elem = soup.find('span', class_='company-name')
            company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
            
            # Extract date
            date_elem = soup.find('time')
            release_date = datetime.now()
            if date_elem:
                date_str = date_elem.get('datetime', '')
                if date_str:
                    release_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Extract content
            content_elem = soup.find('div', class_='news-release-content')
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            # Extract contact info
            contact_elem = soup.find('div', class_='contact-info')
            contact_info = contact_elem.get_text(strip=True) if contact_elem else None
            
            return PressRelease(
                title=title,
                company=company,
                release_date=release_date,
                content=content,
                contact_info=contact_info,
                industry=None,
                tags=[]
            )
            
        except Exception as e:
            logger.error(f"Error parsing PR Newswire detail: {e}")
            return None
    
    def _parse_businesswire_detail(self, soup: BeautifulSoup, url: str) -> Optional[PressRelease]:
        """Parse detailed Business Wire press release."""
        try:
            # Extract title
            title_elem = soup.find('h1', class_='news-title')
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract company
            company_elem = soup.find('span', class_='company')
            company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
            
            # Extract date
            date_elem = soup.find('span', class_='date')
            release_date = datetime.now()
            if date_elem:
                date_str = date_elem.get_text(strip=True)
                try:
                    release_date = datetime.strptime(date_str, '%B %d, %Y')
                except:
                    pass
            
            # Extract content
            content_elem = soup.find('div', class_='news-content')
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            return PressRelease(
                title=title,
                company=company,
                release_date=release_date,
                content=content,
                contact_info=None,
                industry=None,
                tags=[]
            )
            
        except Exception as e:
            logger.error(f"Error parsing Business Wire detail: {e}")
            return None
    
    def _parse_globenewswire_detail(self, soup: BeautifulSoup, url: str) -> Optional[PressRelease]:
        """Parse detailed Globe Newswire press release."""
        try:
            # Extract title
            title_elem = soup.find('h1', class_='title')
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract company
            company_elem = soup.find('span', class_='company')
            company = company_elem.get_text(strip=True) if company_elem else "Unknown company"
            
            # Extract date
            date_elem = soup.find('time')
            release_date = datetime.now()
            if date_elem:
                date_str = date_elem.get('datetime', '')
                if date_str:
                    release_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Extract content
            content_elem = soup.find('div', class_='content')
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            return PressRelease(
                title=title,
                company=company,
                release_date=release_date,
                content=content,
                contact_info=None,
                industry=None,
                tags=[]
            )
            
        except Exception as e:
            logger.error(f"Error parsing Globe Newswire detail: {e}")
            return None
    
    def _parse_generic_detail(self, soup: BeautifulSoup, url: str) -> Optional[PressRelease]:
        """Parse generic press release detail."""
        try:
            # Try to extract basic information
            title = "Press Release"
            company = "Unknown company"
            content = ""
            
            # Look for common title patterns
            title_elem = soup.find('h1') or soup.find('h2') or soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Look for content
            content_elem = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
            if content_elem:
                content = content_elem.get_text(strip=True)
            
            return PressRelease(
                title=title,
                company=company,
                release_date=datetime.now(),
                content=content,
                contact_info=None,
                industry=None,
                tags=[]
            )
            
        except Exception as e:
            logger.error(f"Error parsing generic detail: {e}")
            return None
    
    def convert_to_public_source(self, press_release: PressRelease) -> PublicSource:
        """Convert press release to public source format."""
        return PublicSource(
            id=generate_id(f"press_{press_release.company}_{press_release.release_date.strftime('%Y%m%d')}"),
            title=press_release.title,
            content=press_release.content,
            source_type=SourceType.PRESS_RELEASE,
            published_date=press_release.release_date,
            author=press_release.company,
            organization=press_release.company,
            metadata={
                "company": press_release.company,
                "contact_info": press_release.contact_info,
                "industry": press_release.industry,
                "tags": press_release.tags
            },
            tags=["press_release", "announcement", "news"]
        )


RATE_LIMIT = asyncio.Semaphore(5)


async def _fetch_with_retry(client: httpx.AsyncClient, url: str, params: Dict[str, Any], retries: int = 3) -> httpx.Response:
    for attempt in range(retries):
        try:
            async with RATE_LIMIT:
                response = await client.get(url, params=params)
            response.raise_for_status()
            return response
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)


async def search_press_releases(query: str, max_results: int = 100, sources: List[str] | None = None) -> List[PublicSource]:
    """Asynchronously search press releases and parse HTML into public sources."""
    # PLACEHOLDER endpoint. Returns no results until pointed at a real service
    # (e.g. the PR Newswire / Business Wire sources in PressReleasesCrawler.sources).
    url = "https://example.com/press"
    params = {"q": query, "n": max_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await _fetch_with_retry(client, url, params)
        html = response.text
    except Exception:
        logger.exception("Press release search failed")
        return []

    articles = re.findall(r"<article>(.*?)</article>", html, re.S)
    results: List[PublicSource] = []
    for chunk in articles[:max_results]:
        title_match = re.search(r"<h1>(.*?)</h1>", chunk)
        body_match = re.search(r"<p>(.*?)</p>", chunk)
        link_match = re.search(r"<a href=\"(.*?)\"", chunk)
        date_match = re.search(r"<time>(.*?)</time>", chunk)
        if not (title_match and body_match and link_match):
            continue
        published = None
        if date_match:
            try:
                published = datetime.fromisoformat(date_match.group(1))
            except ValueError:
                published = None
        results.append(
            PublicSource(
                id=generate_id(title_match.group(1)),
                title=title_match.group(1),
                content=body_match.group(1),
                source_type=SourceType.PRESS_RELEASE,
                url=link_match.group(1),
                published_date=published,
                tags=["press", "release"],
            )
        )

    return results
