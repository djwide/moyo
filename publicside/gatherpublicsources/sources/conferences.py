"""Conference talks crawler for gathering presentations, talks, and conference proceedings."""

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

from moyo.publicside.gatherpublicsources.schema import ConferenceTalk, PublicSource, SourceType, SearchQuery
from shared_utils import generate_id

logger = logging.getLogger(__name__)


class ConferenceTalksCrawler:
    """Crawler for conference talks and presentations."""
    
    def __init__(self):
        """Initialize the conference talks crawler."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Conference sources
        self.sources = {
            'arxiv': {
                'base_url': 'https://arxiv.org',
                'search_url': 'https://arxiv.org/search/',
                'api_url': 'http://export.arxiv.org/api/query'
            },
            'ieee': {
                'base_url': 'https://ieeexplore.ieee.org',
                'search_url': 'https://ieeexplore.ieee.org/search/',
                'api_url': 'https://ieeexplore.ieee.org/rest/search'
            },
            'acm': {
                'base_url': 'https://dl.acm.org',
                'search_url': 'https://dl.acm.org/action/doSearch',
                'api_url': 'https://dl.acm.org/api/search'
            },
            'youtube': {
                'base_url': 'https://www.youtube.com',
                'search_url': 'https://www.youtube.com/results',
                'api_url': 'https://www.googleapis.com/youtube/v3/search'
            },
            'slideshare': {
                'base_url': 'https://www.slideshare.net',
                'search_url': 'https://www.slideshare.net/search/',
                'api_url': 'https://www.slideshare.net/api/2/search_slideshows'
            }
        }
        
        # Major conferences
        self.major_conferences = {
            'ai': ['NeurIPS', 'ICML', 'ICLR', 'AAAI', 'IJCAI', 'ACL', 'EMNLP', 'CVPR', 'ICCV', 'ECCV'],
            'systems': ['SIGCOMM', 'NSDI', 'OSDI', 'SOSP', 'ASPLOS', 'ISCA', 'MICRO', 'HPCA'],
            'security': ['CCS', 'S&P', 'USENIX Security', 'NDSS', 'Black Hat', 'DEF CON'],
            'databases': ['SIGMOD', 'VLDB', 'ICDE', 'PODS', 'SIGIR', 'WWW'],
            'networking': ['SIGCOMM', 'INFOCOM', 'NSDI', 'CoNEXT', 'IMC']
        }
    
    def search_conference_talks(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search for conference talks using the given query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of conference talks
        """
        talks = []
        
        try:
            # Search different sources
            sources_to_search = query.filters.get('sources', ['arxiv', 'ieee', 'acm'])
            
            for source in sources_to_search:
                if source in self.sources:
                    source_talks = self._search_source(source, query)
                    talks.extend(source_talks)
            
            logger.info(f"Found {len(talks)} conference talks for query: {query.query}")
            
        except Exception as e:
            logger.error(f"Error searching conference talks: {e}")
        
        return talks[:query.max_results]
    
    def _search_source(self, source: str, query: SearchQuery) -> List[ConferenceTalk]:
        """Search a specific conference source."""
        try:
            if source == 'arxiv':
                return self._search_arxiv(query)
            elif source == 'ieee':
                return self._search_ieee(query)
            elif source == 'acm':
                return self._search_acm(query)
            elif source == 'youtube':
                return self._search_youtube(query)
            elif source == 'slideshare':
                return self._search_slideshare(query)
            else:
                logger.warning(f"Unsupported source: {source}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching {source}: {e}")
            return []
    
    def _search_arxiv(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search arXiv for conference papers and talks."""
        talks = []
        
        try:
            # arXiv search parameters
            search_params = {
                'search_query': f'all:"{query.query}"',
                'start': 0,
                'max_results': min(query.max_results, 100),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['search_query'] += f' AND submittedDate:[{query.date_range["from"].strftime("%Y%m%d")}0000 TO 999912312359]'
                if 'to' in query.date_range:
                    search_params['search_query'] += f' AND submittedDate:[000001010000 TO {query.date_range["to"].strftime("%Y%m%d")}2359]'
            
            # Make search request
            search_url = f"{self.sources['arxiv']['api_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            talks.extend(self._parse_arxiv_results(response.text))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
        
        return talks
    
    def _search_ieee(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search IEEE Xplore for conference papers."""
        talks = []
        
        try:
            # IEEE search parameters
            search_params = {
                'queryText': query.query,
                'highlight': 'true',
                'returnFacets': 'ALL',
                'returnType': 'SEARCH',
                'pageNumber': 1,
                'rowsPerPage': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['ranges'] = f"{query.date_range['from'].strftime('%Y')}_YYYY"
                if 'to' in query.date_range:
                    search_params['ranges'] = f"YYYY_{query.date_range['to'].strftime('%Y')}"
            
            # Make search request
            search_url = f"{self.sources['ieee']['api_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            talks.extend(self._parse_ieee_results(data))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching IEEE: {e}")
        
        return talks
    
    def _search_acm(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search ACM Digital Library for conference papers."""
        talks = []
        
        try:
            # ACM search parameters
            search_params = {
                'AllField': query.query,
                'pageSize': min(query.max_results, 50),
                'startPage': 0
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['AfterYear'] = query.date_range['from'].strftime('%Y')
                if 'to' in query.date_range:
                    search_params['BeforeYear'] = query.date_range['to'].strftime('%Y')
            
            # Make search request
            search_url = f"{self.sources['acm']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            talks.extend(self._parse_acm_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching ACM: {e}")
        
        return talks
    
    def _search_youtube(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search YouTube for conference talks and presentations."""
        talks = []
        
        try:
            # YouTube search parameters
            search_params = {
                'part': 'snippet',
                'q': f"{query.query} conference talk presentation",
                'type': 'video',
                'maxResults': min(query.max_results, 50),
                'order': 'relevance',
                'videoDuration': 'medium'  # Filter for longer videos (talks)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['publishedAfter'] = query.date_range['from'].isoformat() + 'Z'
                if 'to' in query.date_range:
                    search_params['publishedBefore'] = query.date_range['to'].isoformat() + 'Z'
            
            # Make search request
            search_url = f"{self.sources['youtube']['api_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            talks.extend(self._parse_youtube_results(data))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
        
        return talks
    
    def _search_slideshare(self, query: SearchQuery) -> List[ConferenceTalk]:
        """Search SlideShare for conference presentations."""
        talks = []
        
        try:
            # SlideShare search parameters
            search_params = {
                'q': query.query,
                'page': 1,
                'items_per_page': min(query.max_results, 50)
            }
            
            # Make search request
            search_url = f"{self.sources['slideshare']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            talks.extend(self._parse_slideshare_results(soup))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching SlideShare: {e}")
        
        return talks
    
    def _parse_arxiv_results(self, xml_content: str) -> List[ConferenceTalk]:
        """Parse arXiv search results."""
        talks = []
        
        try:
            # Simple XML parsing (in production, use proper XML parser)
            # Look for entry tags
            entries = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL)
            
            for entry in entries:
                try:
                    # Extract title
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    title = title_match.group(1) if title_match else "No title"
                    
                    # Extract authors
                    authors = []
                    author_matches = re.findall(r'<name>(.*?)</name>', entry)
                    authors.extend(author_matches)
                    
                    # Extract abstract
                    abstract_match = re.search(r'<summary>(.*?)</summary>', entry)
                    abstract = abstract_match.group(1) if abstract_match else ""
                    
                    # Extract published date
                    published_match = re.search(r'<published>(.*?)</published>', entry)
                    published_date = datetime.now()
                    if published_match:
                        date_str = published_match.group(1)
                        try:
                            published_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Determine conference from title/abstract
                    conference = self._detect_conference(title + " " + abstract)
                    year = published_date.year
                    
                    talk = ConferenceTalk(
                        title=title,
                        conference=conference,
                        year=year,
                        speakers=authors,
                        abstract=abstract,
                        slides_url=None,
                        video_url=None,
                        paper_url=None,
                        venue=None,
                        track=None
                    )
                    
                    talks.append(talk)
                    
                except Exception as e:
                    logger.error(f"Error parsing arXiv entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing arXiv results: {e}")
        
        return talks
    
    def _parse_ieee_results(self, data: Dict[str, Any]) -> List[ConferenceTalk]:
        """Parse IEEE search results."""
        talks = []
        
        try:
            if 'records' in data:
                for record in data['records']:
                    try:
                        title = record.get('articleTitle', 'No title')
                        authors = record.get('authors', {}).get('authors', [])
                        abstract = record.get('abstract', '')
                        
                        # Extract conference information
                        publication_title = record.get('publicationTitle', '')
                        conference = self._detect_conference(publication_title)
                        
                        # Extract year
                        year = datetime.now().year
                        if 'publicationYear' in record:
                            year = int(record['publicationYear'])
                        
                        # Extract DOI
                        doi = record.get('doi', '')
                        paper_url = f"https://ieeexplore.ieee.org/document/{doi}" if doi else None
                        
                        talk = ConferenceTalk(
                            title=title,
                            conference=conference,
                            year=year,
                            speakers=[author.get('preferredName', '') for author in authors],
                            abstract=abstract,
                            slides_url=None,
                            video_url=None,
                            paper_url=paper_url,
                            venue=None,
                            track=None
                        )
                        
                        talks.append(talk)
                        
                    except Exception as e:
                        logger.error(f"Error parsing IEEE record: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing IEEE results: {e}")
        
        return talks
    
    def _parse_acm_results(self, soup: BeautifulSoup) -> List[ConferenceTalk]:
        """Parse ACM search results."""
        talks = []
        
        try:
            # Look for article entries
            articles = soup.find_all('div', class_='issue-item')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.find('h5', class_='issue-item__title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract authors
                    authors = []
                    author_elems = article.find_all('a', class_='author')
                    for author_elem in author_elems:
                        authors.append(author_elem.get_text(strip=True))
                    
                    # Extract abstract
                    abstract_elem = article.find('div', class_='issue-item__abstract')
                    abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                    
                    # Extract conference
                    conf_elem = article.find('span', class_='epub-section__title')
                    conference = conf_elem.get_text(strip=True) if conf_elem else "Unknown conference"
                    
                    # Extract year
                    year = datetime.now().year
                    year_elem = article.find('span', class_='epub-section__date')
                    if year_elem:
                        year_text = year_elem.get_text(strip=True)
                        year_match = re.search(r'\d{4}', year_text)
                        if year_match:
                            year = int(year_match.group())
                    
                    talk = ConferenceTalk(
                        title=title,
                        conference=conference,
                        year=year,
                        speakers=authors,
                        abstract=abstract,
                        slides_url=None,
                        video_url=None,
                        paper_url=None,
                        venue=None,
                        track=None
                    )
                    
                    talks.append(talk)
                    
                except Exception as e:
                    logger.error(f"Error parsing ACM article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing ACM results: {e}")
        
        return talks
    
    def _parse_youtube_results(self, data: Dict[str, Any]) -> List[ConferenceTalk]:
        """Parse YouTube search results."""
        talks = []
        
        try:
            if 'items' in data:
                for item in data['items']:
                    try:
                        snippet = item.get('snippet', {})
                        title = snippet.get('title', 'No title')
                        description = snippet.get('description', '')
                        
                        # Extract conference from title/description
                        conference = self._detect_conference(title + " " + description)
                        
                        # Extract year from published date
                        published_at = snippet.get('publishedAt', '')
                        year = datetime.now().year
                        if published_at:
                            try:
                                year = datetime.fromisoformat(published_at.replace('Z', '+00:00')).year
                            except:
                                pass
                        
                        # Extract video URL
                        video_id = item.get('id', {}).get('videoId', '')
                        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
                        
                        # Extract channel (speaker)
                        channel_title = snippet.get('channelTitle', '')
                        speakers = [channel_title] if channel_title else []
                        
                        talk = ConferenceTalk(
                            title=title,
                            conference=conference,
                            year=year,
                            speakers=speakers,
                            abstract=description,
                            slides_url=None,
                            video_url=video_url,
                            paper_url=None,
                            venue=None,
                            track=None
                        )
                        
                        talks.append(talk)
                        
                    except Exception as e:
                        logger.error(f"Error parsing YouTube item: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing YouTube results: {e}")
        
        return talks
    
    def _parse_slideshare_results(self, soup: BeautifulSoup) -> List[ConferenceTalk]:
        """Parse SlideShare search results."""
        talks = []
        
        try:
            # Look for presentation entries
            presentations = soup.find_all('div', class_='slide')
            
            for presentation in presentations:
                try:
                    # Extract title
                    title_elem = presentation.find('h3', class_='title')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract author
                    author_elem = presentation.find('span', class_='author')
                    author = author_elem.get_text(strip=True) if author_elem else "Unknown author"
                    
                    # Extract description
                    desc_elem = presentation.find('p', class_='description')
                    description = desc_elem.get_text(strip=True) if desc_elem else ""
                    
                    # Extract slides URL
                    link_elem = presentation.find('a', href=True)
                    slides_url = urljoin(self.sources['slideshare']['base_url'], link_elem['href']) if link_elem else None
                    
                    # Extract conference from title/description
                    conference = self._detect_conference(title + " " + description)
                    
                    talk = ConferenceTalk(
                        title=title,
                        conference=conference,
                        year=datetime.now().year,
                        speakers=[author],
                        abstract=description,
                        slides_url=slides_url,
                        video_url=None,
                        paper_url=None,
                        venue=None,
                        track=None
                    )
                    
                    talks.append(talk)
                    
                except Exception as e:
                    logger.error(f"Error parsing SlideShare presentation: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing SlideShare results: {e}")
        
        return talks
    
    def _detect_conference(self, text: str) -> str:
        """Detect conference name from text."""
        text_lower = text.lower()
        
        # Check major conferences
        for category, conferences in self.major_conferences.items():
            for conference in conferences:
                if conference.lower() in text_lower:
                    return conference
        
        # Look for common conference patterns
        conference_patterns = [
            r'(\w+)\s+\d{4}',  # Conference name + year
            r'(\w+)\s+conference',
            r'(\w+)\s+workshop',
            r'(\w+)\s+summit',
            r'(\w+)\s+meeting'
        ]
        
        for pattern in conference_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).title()
        
        return "Unknown conference"
    
    def search_by_conference(self, conference: str, year: Optional[int] = None, 
                           max_results: int = 100) -> List[ConferenceTalk]:
        """Search for talks from a specific conference.
        
        Args:
            conference: Conference name
            year: Conference year (optional)
            max_results: Maximum number of results
            
        Returns:
            List of conference talks
        """
        talks = []
        
        try:
            # Search across different sources for the specific conference
            query = SearchQuery(
                query=conference,
                source_type=SourceType.CONFERENCE_TALK,
                max_results=max_results
            )
            
            # Search arXiv
            arxiv_talks = self._search_arxiv(query)
            talks.extend([talk for talk in arxiv_talks if conference.lower() in talk.conference.lower()])
            
            # Search IEEE
            ieee_talks = self._search_ieee(query)
            talks.extend([talk for talk in ieee_talks if conference.lower() in talk.conference.lower()])
            
            # Search ACM
            acm_talks = self._search_acm(query)
            talks.extend([talk for talk in acm_talks if conference.lower() in talk.conference.lower()])
            
            # Filter by year if specified
            if year:
                talks = [talk for talk in talks if talk.year == year]
            
            logger.info(f"Found {len(talks)} talks for conference: {conference}")
            
        except Exception as e:
            logger.error(f"Error searching by conference: {e}")
        
        return talks[:max_results]
    
    def convert_to_public_source(self, talk: ConferenceTalk) -> PublicSource:
        """Convert conference talk to public source format."""
        # Create content from talk information
        content = f"Conference: {talk.conference} {talk.year}\n"
        content += f"Title: {talk.title}\n"
        content += f"Speakers: {', '.join(talk.speakers)}\n"
        content += f"Abstract: {talk.abstract}\n"
        
        if talk.venue:
            content += f"Venue: {talk.venue}\n"
        if talk.track:
            content += f"Track: {talk.track}\n"
        
        return PublicSource(
            id=generate_id(f"talk_{talk.conference}_{talk.year}"),
            title=talk.title,
            content=content,
            source_type=SourceType.CONFERENCE_TALK,
            url=talk.video_url or talk.slides_url or talk.paper_url,
            published_date=datetime(talk.year, 1, 1),  # Approximate date
            author=', '.join(talk.speakers),
            organization=talk.conference,
            metadata={
                "conference": talk.conference,
                "year": talk.year,
                "speakers": talk.speakers,
                "venue": talk.venue,
                "track": talk.track,
                "video_url": talk.video_url,
                "slides_url": talk.slides_url,
                "paper_url": talk.paper_url
            },
            tags=["conference", "talk", "presentation", talk.conference.lower()]
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


async def search_conference_talks(query: str, max_results: int = 100, sources: List[str] | None = None) -> List[PublicSource]:
    """Asynchronously search conference talks and return public sources."""
    url = "https://example.com/talks"
    params = {"q": query, "n": max_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await _fetch_with_retry(client, url, params)
        data = response.json()
    except Exception:
        logger.exception("Conference talk search failed")
        return []

    results: List[PublicSource] = []
    for item in data.get("results", [])[:max_results]:
        published = None
        if item.get("date"):
            try:
                published = datetime.fromisoformat(item["date"])
            except ValueError:
                published = None
        results.append(
            PublicSource(
                id=generate_id(item.get("id", "")),
                title=item.get("title", ""),
                content=item.get("abstract", ""),
                source_type=SourceType.CONFERENCE_TALK,
                url=item.get("url"),
                published_date=published,
                tags=["conference"],
            )
        )

    return results
