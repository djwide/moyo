"""Patent crawler for gathering patent information from various patent offices."""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode, quote_plus
import json
import re
import asyncio
import httpx

from moyo.publicside.gatherpublicsources.schema import PatentDocument, PublicSource, SourceType, SearchQuery
from shared_utils import generate_id

logger = logging.getLogger(__name__)


class PatentCrawler:
    """Crawler for patent databases."""
    
    def __init__(self):
        """Initialize the patent crawler."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Patent office APIs and endpoints
        self.patent_offices = {
            'USPTO': {
                'base_url': 'https://patents.google.com',
                'search_url': 'https://patents.google.com/xhr/query',
                'detail_url': 'https://patents.google.com/patent/'
            },
            'EPO': {
                'base_url': 'https://worldwide.espacenet.com',
                'search_url': 'https://worldwide.espacenet.com/patent/search/result',
                'detail_url': 'https://worldwide.espacenet.com/patent/'
            }
        }
    
    def search_patents(self, query: SearchQuery) -> List[PatentDocument]:
        """Search for patents using the given query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of patent documents
        """
        patents = []
        
        try:
            # Search USPTO (Google Patents)
            uspto_patents = self._search_uspto(query)
            patents.extend(uspto_patents)
            
            # Search EPO if specified
            if 'EPO' in query.filters.get('offices', ['USPTO']):
                epo_patents = self._search_epo(query)
                patents.extend(epo_patents)
            
            logger.info(f"Found {len(patents)} patents for query: {query.query}")
            
        except Exception as e:
            logger.error(f"Error searching patents: {e}")
        
        return patents[:query.max_results]
    
    def _search_uspto(self, query: SearchQuery) -> List[PatentDocument]:
        """Search USPTO patents via Google Patents API."""
        patents = []
        
        try:
            # Google Patents search parameters
            search_params = {
                'q': query.query,
                'language': 'ENGLISH',
                'type': 'PATENT',
                'num': min(query.max_results, 100)
            }
            
            # Add date filters if specified
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['after'] = query.date_range['from'].strftime('%Y%m%d')
                if 'to' in query.date_range:
                    search_params['before'] = query.date_range['to'].strftime('%Y%m%d')
            
            # Make search request
            search_url = f"{self.patent_offices['USPTO']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            if 'results' in data:
                for result in data['results']['cluster']:
                    patent = self._parse_uspto_patent(result)
                    if patent:
                        patents.append(patent)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching USPTO: {e}")
        
        return patents
    
    def _search_epo(self, query: SearchQuery) -> List[PatentDocument]:
        """Search EPO patents."""
        patents = []
        
        try:
            # EPO search parameters
            search_params = {
                'q': query.query,
                'lang': 'EN',
                'maxRec': min(query.max_results, 50)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['date'] = f"{query.date_range['from'].strftime('%Y%m%d')}-{query.date_range.get('to', datetime.now()).strftime('%Y%m%d')}"
            
            # Make search request
            search_url = f"{self.patent_offices['EPO']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse EPO results (simplified)
            patents.extend(self._parse_epo_results(response.text))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching EPO: {e}")
        
        return patents
    
    def _parse_uspto_patent(self, patent_data: Dict[str, Any]) -> Optional[PatentDocument]:
        """Parse USPTO patent data from Google Patents."""
        try:
            # Extract basic information
            patent_number = patent_data.get('patent_number', '')
            title = patent_data.get('title', '')
            abstract = patent_data.get('abstract', '')
            
            # Extract inventors
            inventors = []
            if 'inventor' in patent_data:
                for inventor in patent_data['inventor']:
                    inventors.append(inventor.get('name', ''))
            
            # Extract assignee
            assignee = None
            if 'assignee' in patent_data:
                assignee = patent_data['assignee'][0].get('name', '')
            
            # Extract dates
            filing_date = None
            publication_date = None
            
            if 'filing_date' in patent_data:
                filing_date = datetime.strptime(patent_data['filing_date'], '%Y-%m-%d')
            
            if 'publication_date' in patent_data:
                publication_date = datetime.strptime(patent_data['publication_date'], '%Y-%m-%d')
            
            # Extract claims (simplified)
            claims = []
            if 'claims' in patent_data:
                for claim in patent_data['claims']:
                    claims.append(claim.get('text', ''))
            
            return PatentDocument(
                patent_number=patent_number,
                title=title,
                abstract=abstract,
                description=patent_data.get('description', ''),
                claims=claims,
                inventors=inventors,
                assignee=assignee,
                filing_date=filing_date,
                publication_date=publication_date,
                patent_office='USPTO',
                status='published'
            )
            
        except Exception as e:
            logger.error(f"Error parsing USPTO patent: {e}")
            return None
    
    def _parse_epo_results(self, html_content: str) -> List[PatentDocument]:
        """Parse EPO search results from HTML."""
        patents = []
        
        try:
            # Extract patent information from HTML (simplified)
            # This would need more sophisticated HTML parsing in production
            
            # Look for patent numbers
            patent_numbers = re.findall(r'EP\d+[A-Z]?\d+', html_content)
            
            for patent_number in patent_numbers[:10]:  # Limit results
                # Create basic patent document
                patent = PatentDocument(
                    patent_number=patent_number,
                    title=f"EPO Patent {patent_number}",
                    abstract="Abstract not available",
                    description="Description not available",
                    claims=[],
                    inventors=[],
                    patent_office='EPO',
                    status='published'
                )
                patents.append(patent)
        
        except Exception as e:
            logger.error(f"Error parsing EPO results: {e}")
        
        return patents
    
    def get_patent_details(self, patent_number: str, office: str = 'USPTO') -> Optional[PatentDocument]:
        """Get detailed information for a specific patent.
        
        Args:
            patent_number: Patent number
            office: Patent office (USPTO, EPO, etc.)
            
        Returns:
            Detailed patent document
        """
        try:
            if office == 'USPTO':
                return self._get_uspto_patent_details(patent_number)
            elif office == 'EPO':
                return self._get_epo_patent_details(patent_number)
            else:
                logger.warning(f"Unsupported patent office: {office}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting patent details: {e}")
            return None
    
    def _get_uspto_patent_details(self, patent_number: str) -> Optional[PatentDocument]:
        """Get detailed USPTO patent information."""
        try:
            detail_url = f"{self.patent_offices['USPTO']['detail_url']}{patent_number}/en"
            
            response = self.session.get(detail_url)
            response.raise_for_status()
            
            # Parse detailed patent information
            # This would require more sophisticated HTML parsing
            # For now, return a basic structure
            
            return PatentDocument(
                patent_number=patent_number,
                title=f"US Patent {patent_number}",
                abstract="Detailed abstract not available",
                description="Detailed description not available",
                claims=[],
                inventors=[],
                patent_office='USPTO',
                status='published'
            )
            
        except Exception as e:
            logger.error(f"Error getting USPTO patent details: {e}")
            return None
    
    def _get_epo_patent_details(self, patent_number: str) -> Optional[PatentDocument]:
        """Get detailed EPO patent information."""
        try:
            detail_url = f"{self.patent_offices['EPO']['detail_url']}{patent_number}/en"
            
            response = self.session.get(detail_url)
            response.raise_for_status()
            
            # Parse detailed EPO patent information
            return PatentDocument(
                patent_number=patent_number,
                title=f"EP Patent {patent_number}",
                abstract="Detailed abstract not available",
                description="Detailed description not available",
                claims=[],
                inventors=[],
                patent_office='EPO',
                status='published'
            )
            
        except Exception as e:
            logger.error(f"Error getting EPO patent details: {e}")
            return None
    
    def convert_to_public_source(self, patent: PatentDocument) -> PublicSource:
        """Convert patent document to public source format."""
        # Combine title, abstract, and description
        content = f"{patent.title}\n\n{patent.abstract}\n\n{patent.description}"
        
        # Add claims if available
        if patent.claims:
            content += "\n\nClaims:\n" + "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(patent.claims))
        
        return PublicSource(
            id=generate_id(f"patent_{patent.patent_number}"),
            title=patent.title,
            content=content,
            source_type=SourceType.PATENT,
            url=f"{self.patent_offices[patent.patent_office]['detail_url']}{patent.patent_number}",
            published_date=patent.publication_date,
            author=", ".join(patent.inventors) if patent.inventors else None,
            organization=patent.assignee,
            metadata={
                "patent_number": patent.patent_number,
                "patent_office": patent.patent_office,
                "filing_date": patent.filing_date.isoformat() if patent.filing_date else None,
                "classification": patent.classification,
                "status": patent.status
            },
            tags=["patent", patent.patent_office.lower()]
        )


RATE_LIMIT = asyncio.Semaphore(5)


async def _fetch_with_retry(client: httpx.AsyncClient, url: str, params: Dict[str, Any], retries: int = 3) -> httpx.Response:
    """Fetch a URL with simple retry and rate limiting."""
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


async def search_patents(query: str, max_results: int = 100, offices: List[str] | None = None) -> List[PublicSource]:
    """Asynchronously search patents via HTTP and return public sources."""
    url = "https://example.com/patents"
    params = {"q": query, "n": max_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await _fetch_with_retry(client, url, params)
        data = response.json()
    except Exception:
        logger.exception("Patent search failed")
        return []

    sources: List[PublicSource] = []
    for item in data.get("results", [])[:max_results]:
        published = None
        if item.get("date"):
            try:
                published = datetime.fromisoformat(item["date"])
            except ValueError:
                published = None
        sources.append(
            PublicSource(
                id=generate_id(item.get("id", "")),
                title=item.get("title", ""),
                content=item.get("abstract", ""),
                source_type=SourceType.PATENT,
                url=item.get("url"),
                published_date=published,
                metadata={"office": item.get("office")},
                tags=["patent"],
            )
        )

    return sources
