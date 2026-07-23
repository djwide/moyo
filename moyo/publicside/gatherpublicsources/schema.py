"""Schema definitions for public sources crawler."""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Types of public sources."""
    PATENT = "patent"
    PRESS_RELEASE = "press_release"
    GIT_COMMIT = "git_commit"
    CONFERENCE_TALK = "conference_talk"
    LEAKED_CODE = "leaked_code"
    WEB_SEARCH = "web_search"
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"


class CrawlStatus(str, Enum):
    """Status of crawling operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


class PublicSource(BaseModel):
    """Represents a public source document."""
    id: str
    title: str
    content: str
    source_type: SourceType
    url: Optional[HttpUrl] = None
    source_url: Optional[HttpUrl] = None
    published_date: Optional[datetime] = None
    crawled_date: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    organization: Optional[str] = None
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    relevance_score: Optional[float] = None
    confidence_score: Optional[float] = None


class PatentDocument(BaseModel):
    """Patent-specific document structure."""
    patent_number: str
    title: str
    abstract: str
    description: str
    claims: List[str]
    inventors: List[str]
    assignee: Optional[str] = None
    filing_date: Optional[datetime] = None
    publication_date: Optional[datetime] = None
    patent_office: str = "USPTO"
    classification: Optional[str] = None
    status: str = "published"


class GitCommit(BaseModel):
    """Git commit information."""
    commit_hash: str
    repository: str
    author: str
    author_email: str
    commit_date: datetime
    message: str
    files_changed: List[str]
    additions: int
    deletions: int
    branch: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ConferenceTalk(BaseModel):
    """Conference talk information."""
    title: str
    conference: str
    year: int
    speakers: List[str]
    abstract: str
    slides_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None
    paper_url: Optional[HttpUrl] = None
    venue: Optional[str] = None
    track: Optional[str] = None


class PressRelease(BaseModel):
    """Press release information."""
    title: str
    company: str
    release_date: datetime
    content: str
    contact_info: Optional[str] = None
    industry: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class LeakedCode(BaseModel):
    """Leaked code information."""
    title: str
    source: str
    leak_date: Optional[datetime] = None
    discovery_date: datetime = Field(default_factory=datetime.now)
    code_content: str
    language: Optional[str] = None
    repository: Optional[str] = None
    organization: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical


class CrawlConfig(BaseModel):
    """Configuration for crawling operations."""
    topic: str = Field(
        default="",
        description="Primary topic label for jobs and output paths; per-query topics are passed to crawl().",
    )
    source_types: List[SourceType] = Field(default_factory=list)
    max_results_per_source: int = 100
    max_total_results: int = 1000
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    language: str = "en"
    include_metadata: bool = True
    save_raw_data: bool = True
    output_directory: str = "data/public_sources"
    
    # Rate limiting
    requests_per_second: float = 1.0
    delay_between_requests: float = 1.0
    
    # Filtering
    min_relevance_score: float = 0.0
    min_confidence_score: float = 0.0
    
    # Patents
    patent_offices: List[str] = Field(default_factory=lambda: ["USPTO", "EPO", "JPO"])
    patent_statuses: List[str] = Field(default_factory=lambda: ["published", "granted"])
    
    # Git
    git_repositories: List[str] = Field(default_factory=list)
    git_organizations: List[str] = Field(default_factory=list)
    
    # Conferences
    conferences: List[str] = Field(default_factory=list)
    conference_years: List[int] = Field(default_factory=list)
    
    # Press releases
    companies: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)


class CrawlResult(BaseModel):
    """Result of a crawling operation."""
    success: bool
    message: str
    sources_found: int = 0
    sources_processed: int = 0
    sources_failed: int = 0
    processing_time: float = 0.0
    output_path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrawlJob(BaseModel):
    """A crawling job."""
    id: str
    config: CrawlConfig
    status: CrawlStatus = CrawlStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[CrawlResult] = None
    progress: Dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Search query for different sources."""
    query: str
    source_type: SourceType
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = 100
    date_range: Optional[Dict[str, datetime]] = None
