"""Main crawler orchestrator for gathering public sources."""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import asyncio

from .schema import (
    CrawlConfig, CrawlResult, CrawlJob, CrawlStatus, 
    PublicSource, SourceType, SearchQuery
)
from moyo.publicside.gatherpublicsources.sources.patents import search_patents
from moyo.publicside.gatherpublicsources.sources.press_releases import search_press_releases
from moyo.publicside.gatherpublicsources.sources.git_commits import search_git_commits
from moyo.publicside.gatherpublicsources.sources.conferences import search_conference_talks
from moyo.publicside.gatherpublicsources.sources.leaks import search_leaked_code
from shared_utils import ensure_directory, generate_id

logger = logging.getLogger(__name__)


class PublicSourcesCrawler:
    """Main orchestrator for crawling public sources."""
    
    def __init__(self, config: Optional[CrawlConfig] = None):
        """Initialize the public sources crawler.
        
        Args:
            config: Crawling configuration
        """
        self.config = config or CrawlConfig()
        self.jobs: Dict[str, CrawlJob] = {}
        
        # Individual crawler instances are no longer required as search functions handle HTTP
    
    def crawl(self, topic: str, source_types: Optional[List[SourceType]] = None) -> CrawlResult:
        """Orchestrate crawling of public sources for a topic.
        
        Args:
            topic: Topic to search for
            source_types: Types of sources to crawl (if None, use config defaults)
            
        Returns:
            CrawlResult with results and statistics
        """
        start_time = time.time()
        result = CrawlResult(success=False, message="")
        
        try:
            logger.info(f"Starting crawl for topic: {topic}")
            
            # Use provided source types or config defaults
            if source_types is None:
                source_types = self.config.source_types or [
                    SourceType.PATENT,
                    SourceType.PRESS_RELEASE,
                    SourceType.GIT_COMMIT,
                    SourceType.CONFERENCE_TALK,
                    SourceType.LEAKED_CODE
                ]
            
            all_sources = []
            total_found = 0
            total_processed = 0
            total_failed = 0
            
            # Crawl each source type
            for source_type in source_types:
                try:
                    logger.info(f"Crawling {source_type.value} sources...")
                    
                    sources = self._crawl_source_type(topic, source_type)
                    all_sources.extend(sources)
                    
                    total_found += len(sources)
                    total_processed += len(sources)
                    
                    logger.info(f"Found {len(sources)} {source_type.value} sources")
                    
                except Exception as e:
                    logger.error(f"Error crawling {source_type.value}: {e}")
                    total_failed += 1
                    result.errors.append(f"Error crawling {source_type.value}: {str(e)}")
                
                # Rate limiting between source types
                time.sleep(self.config.delay_between_requests)
            
            # Apply filtering and scoring
            filtered_sources = self._filter_and_score_sources(all_sources)
            
            # Save results if requested
            output_path = None
            if self.config.save_raw_data:
                output_path = self._save_results(filtered_sources, topic)
            
            processing_time = time.time() - start_time
            
            result.success = True
            result.message = f"Successfully crawled {len(filtered_sources)} sources for topic: {topic}"
            result.sources_found = total_found
            result.sources_processed = total_processed
            result.sources_failed = total_failed
            result.processing_time = processing_time
            result.output_path = output_path
            # Expose filtered sources in metadata for programmatic use
            result.metadata = {"sources": filtered_sources, "topic": topic}
            
            logger.info(f"Crawl completed: {len(filtered_sources)} sources in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            result.message = f"Error during crawl: {str(e)}"
            result.errors.append(str(e))
        
        return result

    def crawl_with_tokens(
        self,
        tokens: List[str],
        source_types: Optional[List[SourceType]] = None,
    ) -> CrawlResult:
        """Crawl public sources using a list of topic tokens.

        This will generate a set of queries by combining tokens and crawl per query,
        merging and deduplicating results per the configured filters.

        Args:
            tokens: Topic tokens to search for.
            source_types: Types of sources to crawl.

        Returns:
            CrawlResult for the combined crawl.
        """
        if not tokens:
            return CrawlResult(success=False, message="No tokens provided")

        combined_results: List[PublicSource] = []
        total_found = 0
        total_processed = 0
        total_failed = 0

        # Generate simple queries by joining tokens into phrases and single tokens
        # Prefer phrases of 2-3 tokens, then singles
        token_set = list(dict.fromkeys([t.strip() for t in tokens if t and t.strip()]))
        queries: List[str] = []
        # Bigrams
        for i in range(len(token_set) - 1):
            queries.append(f"{token_set[i]} {token_set[i+1]}")
        # Trigrams (limited)
        for i in range(len(token_set) - 2):
            queries.append(f"{token_set[i]} {token_set[i+1]} {token_set[i+2]}")
        # Singles
        queries.extend(token_set)

        # Deduplicate while preserving order
        queries = list(dict.fromkeys(queries))

        for q in queries:
            try:
                res = self.crawl(q, source_types)
                if res.success and res.metadata.get("sources"):
                    combined_results.extend(res.metadata["sources"])  # type: ignore[index]
                total_found += res.sources_found
                total_processed += res.sources_processed
            except Exception as e:
                total_failed += 1
                logger.error(f"Token query crawl failed for '{q}': {e}")

        # Filter/score combined
        filtered_sources = self._filter_and_score_sources(combined_results)
        output_path = None
        if self.config.save_raw_data:
            output_path = self._save_results(filtered_sources, "tokens_query")

        return CrawlResult(
            success=True,
            message=f"Crawled with tokens: {len(filtered_sources)} filtered sources",
            sources_found=total_found,
            sources_processed=total_processed,
            sources_failed=total_failed,
            processing_time=0.0,
            output_path=output_path,
            metadata={"queries": queries, "sources": filtered_sources},
        )
    
    def _crawl_source_type(self, topic: str, source_type: SourceType) -> List[PublicSource]:
        """Crawl a specific source type.
        
        Args:
            topic: Topic to search for
            source_type: Type of source to crawl
            
        Returns:
            List of public sources
        """
        try:
            if source_type == SourceType.PATENT:
                return asyncio.run(
                    search_patents(
                        topic,
                        max_results=self.config.max_results_per_source,
                        offices=self.config.patent_offices,
                    )
                )

            elif source_type == SourceType.PRESS_RELEASE:
                return asyncio.run(
                    search_press_releases(
                        topic,
                        max_results=self.config.max_results_per_source,
                        sources=["prnewswire", "businesswire"],
                    )
                )

            elif source_type == SourceType.GIT_COMMIT:
                return asyncio.run(
                    search_git_commits(
                        topic,
                        max_results=self.config.max_results_per_source,
                        platforms=["github", "gitlab"],
                    )
                )

            elif source_type == SourceType.CONFERENCE_TALK:
                return asyncio.run(
                    search_conference_talks(
                        topic,
                        max_results=self.config.max_results_per_source,
                        sources=["arxiv", "ieee", "acm"],
                    )
                )

            elif source_type == SourceType.LEAKED_CODE:
                return asyncio.run(
                    search_leaked_code(
                        topic,
                        max_results=self.config.max_results_per_source,
                        sources=["github_dorks", "security_forums"],
                    )
                )
            
            else:
                logger.warning(f"Unsupported source type: {source_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error crawling {source_type.value}: {e}")
            return []
    
    def _filter_and_score_sources(self, sources: List[PublicSource]) -> List[PublicSource]:
        """Filter and score sources based on configuration.
        
        Args:
            sources: List of sources to filter
            
        Returns:
            Filtered and scored sources
        """
        filtered_sources = []
        
        for source in sources:
            # Apply relevance scoring
            relevance_score = self._calculate_relevance_score(source)
            source.relevance_score = relevance_score
            
            # Apply confidence scoring
            confidence_score = self._calculate_confidence_score(source)
            source.confidence_score = confidence_score
            
            # Filter based on minimum scores
            if (relevance_score >= self.config.min_relevance_score and 
                confidence_score >= self.config.min_confidence_score):
                filtered_sources.append(source)
        
        # Sort by relevance score (descending)
        filtered_sources.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        # Limit to maximum total results
        filtered_sources = filtered_sources[:self.config.max_total_results]
        
        logger.info(f"Filtered {len(sources)} sources to {len(filtered_sources)}")
        
        return filtered_sources
    
    def _calculate_relevance_score(self, source: PublicSource) -> float:
        """Calculate relevance score for a source.
        
        Args:
            source: Source to score
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Base score based on source type
        type_scores = {
            SourceType.PATENT: 0.8,
            SourceType.PRESS_RELEASE: 0.6,
            SourceType.GIT_COMMIT: 0.7,
            SourceType.CONFERENCE_TALK: 0.9,
            SourceType.LEAKED_CODE: 0.5
        }
        
        score += type_scores.get(source.source_type, 0.5)
        
        # Boost score for recent sources
        if source.published_date:
            days_old = (datetime.now() - source.published_date).days
            if days_old <= 30:
                score += 0.1
            elif days_old <= 90:
                score += 0.05
        
        # Boost score for sources with URLs
        if source.url:
            score += 0.05
        
        # Boost score for sources with metadata
        if source.metadata:
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, source: PublicSource) -> float:
        """Calculate confidence score for a source.
        
        Args:
            source: Source to score
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.5  # Base confidence
        
        # Boost confidence for sources with more metadata
        if source.author:
            score += 0.1
        
        if source.organization:
            score += 0.1
        
        if source.url:
            score += 0.1
        
        if source.metadata:
            score += 0.1
        
        # Boost confidence for longer content
        if len(source.content) > 1000:
            score += 0.1
        elif len(source.content) > 500:
            score += 0.05
        
        return min(score, 1.0)
    
    def _save_results(self, sources: List[PublicSource], topic: str) -> str:
        """Save crawl results to disk.
        
        Args:
            sources: List of sources to save
            topic: Topic that was crawled
            
        Returns:
            Path where results were saved
        """
        try:
            # Create output directory
            output_dir = Path(self.config.output_directory)
            ensure_directory(output_dir)
            
            # Create topic-specific directory
            topic_dir = output_dir / topic.replace(' ', '_').lower()
            ensure_directory(topic_dir)
            
            # Save sources as JSON
            sources_file = topic_dir / "sources.json"
            sources_data = [source.dict() for source in sources]
            
            with open(sources_file, 'w', encoding='utf-8') as f:
                json.dump(sources_data, f, indent=2, default=str)
            
            # Save summary
            summary_file = topic_dir / "summary.json"
            summary = {
                "topic": topic,
                "crawl_date": datetime.now().isoformat(),
                "total_sources": len(sources),
                "source_types": list(set(source.source_type.value for source in sources)),
                "date_range": {
                    "earliest": min(source.published_date.isoformat() for source in sources if source.published_date).isoformat(),
                    "latest": max(source.published_date.isoformat() for source in sources if source.published_date).isoformat()
                } if any(source.published_date for source in sources) else None,
                "organizations": list(set(source.organization for source in sources if source.organization)),
                "average_relevance": sum(source.relevance_score or 0 for source in sources) / len(sources) if sources else 0,
                "average_confidence": sum(source.confidence_score or 0 for source in sources) / len(sources) if sources else 0
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {topic_dir}")
            return str(topic_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
    
    def create_job(self, config: CrawlConfig) -> str:
        """Create a new crawling job.
        
        Args:
            config: Crawling configuration
            
        Returns:
            Job ID
        """
        job_id = generate_id("crawl_job")
        
        job = CrawlJob(
            id=job_id,
            config=config,
            status=CrawlStatus.PENDING
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created crawl job: {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[CrawlJob]:
        """Get the status of a crawling job.
        
        Args:
            job_id: Job ID
            
        Returns:
            CrawlJob or None if not found
        """
        return self.jobs.get(job_id)
    
    def run_job(self, job_id: str) -> CrawlResult:
        """Run a crawling job.
        
        Args:
            job_id: Job ID
            
        Returns:
            CrawlResult
        """
        if job_id not in self.jobs:
            return CrawlResult(
                success=False,
                message=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        job.status = CrawlStatus.IN_PROGRESS
        job.started_at = datetime.now()
        
        try:
            # Run the crawl
            result = self.crawl(job.config.topic, job.config.source_types)
            
            # Update job status
            job.status = CrawlStatus.COMPLETED if result.success else CrawlStatus.FAILED
            job.completed_at = datetime.now()
            job.result = result
            
            return result
            
        except Exception as e:
            job.status = CrawlStatus.FAILED
            job.completed_at = datetime.now()
            job.result = CrawlResult(
                success=False,
                message=f"Job failed: {str(e)}"
            )
            
            return job.result
    
    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get progress information for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Progress information
        """
        job = self.get_job_status(job_id)
        if not job:
            return {"error": "Job not found"}
        
        progress = {
            "job_id": job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
        
        if job.result:
            progress.update({
                "sources_found": job.result.sources_found,
                "sources_processed": job.result.sources_processed,
                "processing_time": job.result.processing_time,
                "success": job.result.success
            })
        
        return progress


def crawl(topic: str, source_types: Optional[List[SourceType]] = None, 
          max_results: int = 100) -> CrawlResult:
    """Convenience function to crawl public sources.
    
    Args:
        topic: Topic to search for
        source_types: Types of sources to crawl
        max_results: Maximum number of results per source
        
    Returns:
        CrawlResult with results and statistics
    """
    config = CrawlConfig(
        topic=topic,
        source_types=source_types,
        max_results_per_source=max_results,
        max_total_results=max_results * 5  # Allow more total results
    )
    
    crawler = PublicSourcesCrawler(config)
    return crawler.crawl(topic, source_types)


def crawl_all_sources(topic: str, max_results: int = 100) -> CrawlResult:
    """Crawl all available source types for a topic.
    
    Args:
        topic: Topic to search for
        max_results: Maximum number of results per source
        
    Returns:
        CrawlResult with results and statistics
    """
    all_source_types = [
        SourceType.PATENT,
        SourceType.PRESS_RELEASE,
        SourceType.GIT_COMMIT,
        SourceType.CONFERENCE_TALK,
        SourceType.LEAKED_CODE
    ]
    
    return crawl(topic, all_source_types, max_results)
