"""Leaked code crawler for gathering leaked source code and security vulnerabilities."""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode, urljoin
import json
import re
import asyncio
import httpx

from moyo.publicside.gatherpublicsources.schema import LeakedCode, PublicSource, SourceType, SearchQuery
from shared_utils import generate_id

logger = logging.getLogger(__name__)


class LeakedCodeCrawler:
    """Crawler for leaked source code and security vulnerabilities."""
    
    def __init__(self):
        """Initialize the leaked code crawler."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Leaked code sources (note: these are hypothetical for demonstration)
        self.sources = {
            'github_dorks': {
                'base_url': 'https://github.com',
                'search_url': 'https://github.com/search',
                'api_url': 'https://api.github.com/search/code'
            },
            'pastebin': {
                'base_url': 'https://pastebin.com',
                'search_url': 'https://pastebin.com/search',
                'api_url': 'https://scrape.pastebin.com/api'
            },
            'ghostbin': {
                'base_url': 'https://ghostbin.co',
                'search_url': 'https://ghostbin.co/search',
                'api_url': 'https://ghostbin.co/api'
            },
            'security_forums': {
                'base_url': 'https://security.stackexchange.com',
                'search_url': 'https://security.stackexchange.com/search',
                'api_url': 'https://api.stackexchange.com/2.3/search'
            }
        }
        
        # Common GitHub dorks for finding leaked code
        self.github_dorks = [
            'password',
            'api_key',
            'secret',
            'token',
            'credential',
            'private_key',
            'ssh_key',
            'database_url',
            'connection_string',
            'config',
            'env',
            '.env',
            'credentials.json',
            'secrets.json'
        ]
    
    def search_leaked_code(self, query: SearchQuery) -> List[LeakedCode]:
        """Search for leaked code using the given query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of leaked code items
        """
        leaked_items = []
        
        try:
            # Search different sources
            sources_to_search = query.filters.get('sources', ['github_dorks', 'security_forums'])
            
            for source in sources_to_search:
                if source in self.sources:
                    source_items = self._search_source(source, query)
                    leaked_items.extend(source_items)
            
            logger.info(f"Found {len(leaked_items)} leaked code items for query: {query.query}")
            
        except Exception as e:
            logger.error(f"Error searching leaked code: {e}")
        
        return leaked_items[:query.max_results]
    
    def _search_source(self, source: str, query: SearchQuery) -> List[LeakedCode]:
        """Search a specific leaked code source."""
        try:
            if source == 'github_dorks':
                return self._search_github_dorks(query)
            elif source == 'pastebin':
                return self._search_pastebin(query)
            elif source == 'ghostbin':
                return self._search_ghostbin(query)
            elif source == 'security_forums':
                return self._search_security_forums(query)
            else:
                logger.warning(f"Unsupported source: {source}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching {source}: {e}")
            return []
    
    def _search_github_dorks(self, query: SearchQuery) -> List[LeakedCode]:
        """Search GitHub for potentially leaked code using dorks."""
        leaked_items = []
        
        try:
            # Combine query with common dorks
            search_terms = [query.query]
            if 'dorks' in query.filters:
                search_terms.extend(query.filters['dorks'])
            else:
                search_terms.extend(self.github_dorks[:3])  # Limit to first 3 dorks
            
            for term in search_terms:
                try:
                    # GitHub code search parameters
                    search_params = {
                        'q': f'"{term}"',
                        'type': 'code',
                        'sort': 'indexed',
                        'order': 'desc',
                        'per_page': min(query.max_results // len(search_terms), 30)
                    }
                    
                    # Add file type filters
                    if 'file_types' in query.filters:
                        file_types = query.filters['file_types']
                        if file_types:
                            file_filter = " OR ".join([f'filename:{ft}' for ft in file_types])
                            search_params['q'] += f" ({file_filter})"
                    
                    # Add language filters
                    if 'languages' in query.filters:
                        languages = query.filters['languages']
                        if languages:
                            lang_filter = " OR ".join([f'language:{lang}' for lang in languages])
                            search_params['q'] += f" ({lang_filter})"
                    
                    # Make search request
                    search_url = f"{self.sources['github_dorks']['search_url']}?{urlencode(search_params)}"
                    
                    response = self.session.get(search_url)
                    response.raise_for_status()
                    
                    # Parse search results
                    soup = BeautifulSoup(response.text, 'html.parser')
                    leaked_items.extend(self._parse_github_dorks_results(soup, term))
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error searching GitHub dork '{term}': {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error searching GitHub dorks: {e}")
        
        return leaked_items
    
    def _search_pastebin(self, query: SearchQuery) -> List[LeakedCode]:
        """Search Pastebin for leaked code."""
        leaked_items = []
        
        try:
            # Pastebin search parameters
            search_params = {
                'q': query.query,
                'page': 1,
                'limit': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['from'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['to'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Make search request
            search_url = f"{self.sources['pastebin']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            leaked_items.extend(self._parse_pastebin_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching Pastebin: {e}")
        
        return leaked_items
    
    def _search_ghostbin(self, query: SearchQuery) -> List[LeakedCode]:
        """Search Ghostbin for leaked code."""
        leaked_items = []
        
        try:
            # Ghostbin search parameters
            search_params = {
                'q': query.query,
                'page': 1,
                'limit': min(query.max_results, 50)
            }
            
            # Make search request
            search_url = f"{self.sources['ghostbin']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            leaked_items.extend(self._parse_ghostbin_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching Ghostbin: {e}")
        
        return leaked_items
    
    def _search_security_forums(self, query: SearchQuery) -> List[LeakedCode]:
        """Search security forums for leaked code discussions."""
        leaked_items = []
        
        try:
            # Stack Exchange search parameters
            search_params = {
                'order': 'desc',
                'sort': 'activity',
                'tagged': 'leak;code;security',
                'intitle': query.query,
                'site': 'security',
                'pagesize': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['fromdate'] = int(query.date_range['from'].timestamp())
                if 'to' in query.date_range:
                    search_params['todate'] = int(query.date_range['to'].timestamp())
            
            # Make search request
            search_url = f"{self.sources['security_forums']['api_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            leaked_items.extend(self._parse_security_forums_results(data))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching security forums: {e}")
        
        return leaked_items
    
    def _parse_github_dorks_results(self, soup: BeautifulSoup, search_term: str) -> List[LeakedCode]:
        """Parse GitHub dorks search results."""
        leaked_items = []
        
        try:
            # Look for code search results
            code_items = soup.find_all('div', class_='code-list-item')
            
            for item in code_items:
                try:
                    # Extract repository information
                    repo_elem = item.find('a', class_='js-navigation-open')
                    repository = repo_elem.get_text(strip=True) if repo_elem else "Unknown repository"
                    
                    # Extract file path
                    file_elem = item.find('a', class_='js-navigation-open')
                    file_path = file_elem.get('title', '') if file_elem else ""
                    
                    # Extract code snippet
                    code_elem = item.find('div', class_='highlight')
                    code_content = code_elem.get_text(strip=True) if code_elem else ""
                    
                    # Extract language
                    lang_elem = item.find('span', class_='language')
                    language = lang_elem.get_text(strip=True) if lang_elem else None
                    
                    # Determine severity based on search term
                    severity = self._determine_severity(search_term)
                    
                    leaked_code = LeakedCode(
                        title=f"Leaked {search_term} in {repository}",
                        source="GitHub",
                        leak_date=None,
                        discovery_date=datetime.now(),
                        code_content=code_content,
                        language=language,
                        repository=repository,
                        organization=repository.split('/')[0] if '/' in repository else None,
                        severity=severity
                    )
                    
                    leaked_items.append(leaked_code)
                    
                except Exception as e:
                    logger.error(f"Error parsing GitHub dork item: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing GitHub dorks results: {e}")
        
        return leaked_items
    
    def _parse_pastebin_results(self, soup: BeautifulSoup) -> List[LeakedCode]:
        """Parse Pastebin search results."""
        leaked_items = []
        
        try:
            # Look for paste entries
            paste_items = soup.find_all('div', class_='paste-item')
            
            for item in paste_items:
                try:
                    # Extract title
                    title_elem = item.find('h3', class_='paste-title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract content preview
                    content_elem = item.find('div', class_='paste-content')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract date
                    date_elem = item.find('span', class_='paste-date')
                    leak_date = None
                    if date_elem:
                        date_str = date_elem.get_text(strip=True)
                        try:
                            leak_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    
                    # Extract language
                    lang_elem = item.find('span', class_='paste-language')
                    language = lang_elem.get_text(strip=True) if lang_elem else None
                    
                    leaked_code = LeakedCode(
                        title=title,
                        source="Pastebin",
                        leak_date=leak_date,
                        discovery_date=datetime.now(),
                        code_content=content,
                        language=language,
                        repository=None,
                        organization=None,
                        severity="medium"
                    )
                    
                    leaked_items.append(leaked_code)
                    
                except Exception as e:
                    logger.error(f"Error parsing Pastebin item: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Pastebin results: {e}")
        
        return leaked_items
    
    def _parse_ghostbin_results(self, soup: BeautifulSoup) -> List[LeakedCode]:
        """Parse Ghostbin search results."""
        leaked_items = []
        
        try:
            # Look for paste entries
            paste_items = soup.find_all('div', class_='paste')
            
            for item in paste_items:
                try:
                    # Extract title
                    title_elem = item.find('h3', class_='title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract content preview
                    content_elem = item.find('div', class_='content')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract date
                    date_elem = item.find('span', class_='date')
                    leak_date = None
                    if date_elem:
                        date_str = date_elem.get_text(strip=True)
                        try:
                            leak_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    
                    leaked_code = LeakedCode(
                        title=title,
                        source="Ghostbin",
                        leak_date=leak_date,
                        discovery_date=datetime.now(),
                        code_content=content,
                        language=None,
                        repository=None,
                        organization=None,
                        severity="medium"
                    )
                    
                    leaked_items.append(leaked_code)
                    
                except Exception as e:
                    logger.error(f"Error parsing Ghostbin item: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Ghostbin results: {e}")
        
        return leaked_items
    
    def _parse_security_forums_results(self, data: Dict[str, Any]) -> List[LeakedCode]:
        """Parse security forums search results."""
        leaked_items = []
        
        try:
            if 'items' in data:
                for item in data['items']:
                    try:
                        title = item.get('title', 'No title')
                        body = item.get('body', '')
                        
                        # Extract code blocks from body
                        code_blocks = re.findall(r'<code>(.*?)</code>', body, re.DOTALL)
                        code_content = '\n'.join(code_blocks) if code_blocks else body[:1000]
                        
                        # Extract creation date
                        creation_date = datetime.fromtimestamp(item.get('creation_date', 0))
                        
                        leaked_code = LeakedCode(
                            title=title,
                            source="Security Forum",
                            leak_date=creation_date,
                            discovery_date=datetime.now(),
                            code_content=code_content,
                            language=None,
                            repository=None,
                            organization=None,
                            severity="low"
                        )
                        
                        leaked_items.append(leaked_code)
                        
                    except Exception as e:
                        logger.error(f"Error parsing security forum item: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing security forums results: {e}")
        
        return leaked_items
    
    def _determine_severity(self, search_term: str) -> str:
        """Determine severity level based on search term."""
        high_severity_terms = ['password', 'api_key', 'secret', 'token', 'private_key', 'ssh_key']
        medium_severity_terms = ['credential', 'database_url', 'connection_string', 'config']
        low_severity_terms = ['env', '.env', 'credentials.json', 'secrets.json']
        
        search_term_lower = search_term.lower()
        
        if any(term in search_term_lower for term in high_severity_terms):
            return "high"
        elif any(term in search_term_lower for term in medium_severity_terms):
            return "medium"
        elif any(term in search_term_lower for term in low_severity_terms):
            return "low"
        else:
            return "medium"
    
    def search_by_organization(self, organization: str, max_results: int = 100) -> List[LeakedCode]:
        """Search for leaked code from a specific organization.
        
        Args:
            organization: Organization name
            max_results: Maximum number of results
            
        Returns:
            List of leaked code items
        """
        leaked_items = []
        
        try:
            # Search GitHub for organization-specific leaks
            query = SearchQuery(
                query=f'org:{organization}',
                source_type=SourceType.LEAKED_CODE,
                max_results=max_results,
                filters={'dorks': self.github_dorks[:5]}
            )
            
            github_items = self._search_github_dorks(query)
            leaked_items.extend(github_items)
            
            # Search security forums for organization mentions
            forum_query = SearchQuery(
                query=organization,
                source_type=SourceType.LEAKED_CODE,
                max_results=max_results // 2,
                filters={'sources': ['security_forums']}
            )
            
            forum_items = self._search_security_forums(forum_query)
            leaked_items.extend(forum_items)
            
            logger.info(f"Found {len(leaked_items)} leaked code items for organization: {organization}")
            
        except Exception as e:
            logger.error(f"Error searching by organization: {e}")
        
        return leaked_items[:max_results]
    
    def search_by_language(self, language: str, max_results: int = 100) -> List[LeakedCode]:
        """Search for leaked code in a specific programming language.
        
        Args:
            language: Programming language
            max_results: Maximum number of results
            
        Returns:
            List of leaked code items
        """
        leaked_items = []
        
        try:
            # Search GitHub for language-specific leaks
            query = SearchQuery(
                query=f'language:{language}',
                source_type=SourceType.LEAKED_CODE,
                max_results=max_results,
                filters={'dorks': self.github_dorks[:3], 'languages': [language]}
            )
            
            github_items = self._search_github_dorks(query)
            leaked_items.extend(github_items)
            
            logger.info(f"Found {len(leaked_items)} leaked code items for language: {language}")
            
        except Exception as e:
            logger.error(f"Error searching by language: {e}")
        
        return leaked_items[:max_results]
    
    def convert_to_public_source(self, leaked_code: LeakedCode) -> PublicSource:
        """Convert leaked code to public source format."""
        # Create content from leaked code information
        content = f"Source: {leaked_code.source}\n"
        content += f"Severity: {leaked_code.severity}\n"
        content += f"Discovery Date: {leaked_code.discovery_date.isoformat()}\n"
        
        if leaked_code.leak_date:
            content += f"Leak Date: {leaked_code.leak_date.isoformat()}\n"
        
        if leaked_code.repository:
            content += f"Repository: {leaked_code.repository}\n"
        
        if leaked_code.language:
            content += f"Language: {leaked_code.language}\n"
        
        content += f"\nCode Content:\n{leaked_code.code_content}"
        
        return PublicSource(
            id=generate_id(f"leak_{leaked_code.source}_{leaked_code.discovery_date.strftime('%Y%m%d')}"),
            title=leaked_code.title,
            content=content,
            source_type=SourceType.LEAKED_CODE,
            published_date=leaked_code.leak_date or leaked_code.discovery_date,
            author=leaked_code.organization or "Unknown",
            organization=leaked_code.organization,
            metadata={
                "source": leaked_code.source,
                "severity": leaked_code.severity,
                "language": leaked_code.language,
                "repository": leaked_code.repository,
                "leak_date": leaked_code.leak_date.isoformat() if leaked_code.leak_date else None,
                "discovery_date": leaked_code.discovery_date.isoformat()
            },
            tags=["leaked_code", "security", leaked_code.severity, leaked_code.source.lower()]
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


async def search_leaked_code(query: str, max_results: int = 100, sources: List[str] | None = None) -> List[PublicSource]:
    """Asynchronously search for leaked code snippets."""
    url = "https://example.com/leaks"
    params = {"q": query, "n": max_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await _fetch_with_retry(client, url, params)
        html = response.text
    except Exception:
        logger.exception("Leaked code search failed")
        return []

    pre_blocks = re.findall(r"<pre([^>]*)>(.*?)</pre>", html, re.S)
    results: List[PublicSource] = []
    for attrs, body in pre_blocks[:max_results]:
        title_match = re.search(r'data-title="(.*?)"', attrs)
        url_match = re.search(r'data-url="(.*?)"', attrs)
        date_match = re.search(r'data-date="(.*?)"', attrs)
        content = re.sub(r'<[^>]+>', '', body)
        if not (title_match and url_match):
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
                content=content.strip(),
                source_type=SourceType.LEAKED_CODE,
                url=url_match.group(1),
                published_date=published,
                tags=["leak"],
            )
        )

    return results
