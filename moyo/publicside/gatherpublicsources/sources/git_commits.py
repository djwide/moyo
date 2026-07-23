"""Git commits crawler for gathering code changes and commits from various repositories."""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from urllib.parse import urlencode
import json
import re
import asyncio
import httpx

from moyo.publicside.gatherpublicsources.schema import GitCommit, PublicSource, SourceType, SearchQuery
from shared_utils import generate_id

logger = logging.getLogger(__name__)


class GitCommitsCrawler:
    """Crawler for Git repositories and commits."""
    
    def __init__(self, github_token: Optional[str] = None, gitlab_token: Optional[str] = None):
        """Initialize the Git commits crawler.
        
        Args:
            github_token: GitHub API token for authenticated requests
            gitlab_token: GitLab API token for authenticated requests
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # API tokens
        self.github_token = github_token
        self.gitlab_token = gitlab_token
        
        if github_token:
            self.session.headers.update({
                'Authorization': f'token {github_token}'
            })
        
        # API endpoints
        self.apis = {
            'github': {
                'base_url': 'https://api.github.com',
                'search_url': 'https://api.github.com/search/commits',
                'repo_url': 'https://api.github.com/repos'
            },
            'gitlab': {
                'base_url': 'https://gitlab.com/api/v4',
                'search_url': 'https://gitlab.com/api/v4/search',
                'repo_url': 'https://gitlab.com/api/v4/projects'
            }
        }
    
    def search_commits(self, query: SearchQuery) -> List[GitCommit]:
        """Search for Git commits using the given query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of Git commits
        """
        commits = []
        
        try:
            # Search GitHub
            github_commits = self._search_github(query)
            commits.extend(github_commits)
            
            # Search GitLab if specified
            if 'gitlab' in query.filters.get('platforms', ['github']):
                gitlab_commits = self._search_gitlab(query)
                commits.extend(gitlab_commits)
            
            logger.info(f"Found {len(commits)} commits for query: {query.query}")
            
        except Exception as e:
            logger.error(f"Error searching commits: {e}")
        
        return commits[:query.max_results]
    
    def _search_github(self, query: SearchQuery) -> List[GitCommit]:
        """Search GitHub commits."""
        commits = []
        
        try:
            # GitHub search parameters
            search_params = {
                'q': query.query,
                'sort': 'committer-date',
                'order': 'desc',
                'per_page': min(query.max_results, 100)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['q'] += f" committer-date:>={query.date_range['from'].strftime('%Y-%m-%d')}"
                if 'to' in query.date_range:
                    search_params['q'] += f" committer-date:<={query.date_range['to'].strftime('%Y-%m-%d')}"
            
            # Add repository filters
            if 'repositories' in query.filters:
                repos = query.filters['repositories']
                if repos:
                    repo_filter = " ".join([f"repo:{repo}" for repo in repos])
                    search_params['q'] += f" {repo_filter}"
            
            # Add organization filters
            if 'organizations' in query.filters:
                orgs = query.filters['organizations']
                if orgs:
                    org_filter = " ".join([f"org:{org}" for org in orgs])
                    search_params['q'] += f" {org_filter}"
            
            # Make search request
            search_url = f"{self.apis['github']['search_url']}?{urlencode(search_params)}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    commit = self._parse_github_commit(item)
                    if commit:
                        commits.append(commit)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
        
        return commits
    
    def _search_gitlab(self, query: SearchQuery) -> List[GitCommit]:
        """Search GitLab commits."""
        commits = []
        
        try:
            # GitLab search parameters
            search_params = {
                'scope': 'commits',
                'search': query.query,
                'per_page': min(query.max_results, 100)
            }
            
            # Add date filters
            if query.date_range:
                if 'from' in query.date_range:
                    search_params['after'] = query.date_range['from'].strftime('%Y-%m-%d')
                if 'to' in query.date_range:
                    search_params['before'] = query.date_range['to'].strftime('%Y-%m-%d')
            
            # Make search request
            search_url = f"{self.apis['gitlab']['search_url']}?{urlencode(search_params)}"
            
            if self.gitlab_token:
                search_url += f"&private_token={self.gitlab_token}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            # Parse search results
            data = response.json()
            for item in data:
                commit = self._parse_gitlab_commit(item)
                if commit:
                    commits.append(commit)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching GitLab: {e}")
        
        return commits
    
    def _parse_github_commit(self, commit_data: Dict[str, Any]) -> Optional[GitCommit]:
        """Parse GitHub commit data."""
        try:
            # Extract basic information
            commit_hash = commit_data.get('sha', '')
            message = commit_data.get('commit', {}).get('message', '')
            
            # Extract repository information
            repo_full_name = commit_data.get('repository', {}).get('full_name', '')
            
            # Extract author information
            author = commit_data.get('commit', {}).get('author', {}).get('name', '')
            author_email = commit_data.get('commit', {}).get('author', {}).get('email', '')
            
            # Extract commit date
            commit_date_str = commit_data.get('commit', {}).get('author', {}).get('date', '')
            commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
            
            # Extract files changed
            files_changed = []
            if 'files' in commit_data:
                for file_info in commit_data['files']:
                    files_changed.append(file_info.get('filename', ''))
            
            # Extract statistics
            additions = commit_data.get('stats', {}).get('additions', 0)
            deletions = commit_data.get('stats', {}).get('deletions', 0)
            
            return GitCommit(
                commit_hash=commit_hash,
                repository=repo_full_name,
                author=author,
                author_email=author_email,
                commit_date=commit_date,
                message=message,
                files_changed=files_changed,
                additions=additions,
                deletions=deletions
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitHub commit: {e}")
            return None
    
    def _parse_gitlab_commit(self, commit_data: Dict[str, Any]) -> Optional[GitCommit]:
        """Parse GitLab commit data."""
        try:
            # Extract basic information
            commit_hash = commit_data.get('id', '')
            message = commit_data.get('message', '')
            
            # Extract repository information
            repo_name = commit_data.get('project_id', '')
            
            # Extract author information
            author = commit_data.get('author_name', '')
            author_email = commit_data.get('author_email', '')
            
            # Extract commit date
            commit_date_str = commit_data.get('created_at', '')
            commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
            
            # Extract files changed (simplified)
            files_changed = []
            if 'diff' in commit_data:
                for diff in commit_data['diff']:
                    files_changed.append(diff.get('new_path', ''))
            
            return GitCommit(
                commit_hash=commit_hash,
                repository=str(repo_name),
                author=author,
                author_email=author_email,
                commit_date=commit_date,
                message=message,
                files_changed=files_changed,
                additions=0,  # GitLab doesn't provide this in search results
                deletions=0
            )
            
        except Exception as e:
            logger.error(f"Error parsing GitLab commit: {e}")
            return None
    
    def get_commit_details(self, repository: str, commit_hash: str, platform: str = 'github') -> Optional[GitCommit]:
        """Get detailed information for a specific commit.
        
        Args:
            repository: Repository name (owner/repo for GitHub, project_id for GitLab)
            commit_hash: Commit hash
            platform: Platform (github, gitlab)
            
        Returns:
            Detailed commit information
        """
        try:
            if platform == 'github':
                return self._get_github_commit_details(repository, commit_hash)
            elif platform == 'gitlab':
                return self._get_gitlab_commit_details(repository, commit_hash)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting commit details: {e}")
            return None
    
    def _get_github_commit_details(self, repository: str, commit_hash: str) -> Optional[GitCommit]:
        """Get detailed GitHub commit information."""
        try:
            detail_url = f"{self.apis['github']['repo_url']}/{repository}/commits/{commit_hash}"
            
            response = self.session.get(detail_url)
            response.raise_for_status()
            
            commit_data = response.json()
            return self._parse_github_commit(commit_data)
            
        except Exception as e:
            logger.error(f"Error getting GitHub commit details: {e}")
            return None
    
    def _get_gitlab_commit_details(self, project_id: str, commit_hash: str) -> Optional[GitCommit]:
        """Get detailed GitLab commit information."""
        try:
            detail_url = f"{self.apis['gitlab']['repo_url']}/{project_id}/repository/commits/{commit_hash}"
            
            if self.gitlab_token:
                detail_url += f"?private_token={self.gitlab_token}"
            
            response = self.session.get(detail_url)
            response.raise_for_status()
            
            commit_data = response.json()
            return self._parse_gitlab_commit(commit_data)
            
        except Exception as e:
            logger.error(f"Error getting GitLab commit details: {e}")
            return None
    
    def search_repository_commits(self, repository: str, query: str = "", platform: str = 'github', 
                                 max_results: int = 100) -> List[GitCommit]:
        """Search commits in a specific repository.
        
        Args:
            repository: Repository name
            query: Search query
            platform: Platform (github, gitlab)
            max_results: Maximum number of results
            
        Returns:
            List of commits
        """
        try:
            if platform == 'github':
                return self._search_github_repo_commits(repository, query, max_results)
            elif platform == 'gitlab':
                return self._search_gitlab_repo_commits(repository, query, max_results)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching repository commits: {e}")
            return []
    
    def _search_github_repo_commits(self, repository: str, query: str, max_results: int) -> List[GitCommit]:
        """Search commits in a GitHub repository."""
        commits = []
        
        try:
            # GitHub repository commits API
            commits_url = f"{self.apis['github']['repo_url']}/{repository}/commits"
            
            params = {
                'per_page': min(max_results, 100)
            }
            
            if query:
                params['q'] = query
            
            response = self.session.get(commits_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            for commit_data in data:
                commit = self._parse_github_commit(commit_data)
                if commit:
                    commits.append(commit)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching GitHub repository commits: {e}")
        
        return commits
    
    def _search_gitlab_repo_commits(self, project_id: str, query: str, max_results: int) -> List[GitCommit]:
        """Search commits in a GitLab repository."""
        commits = []
        
        try:
            # GitLab repository commits API
            commits_url = f"{self.apis['gitlab']['repo_url']}/{project_id}/repository/commits"
            
            params = {
                'per_page': min(max_results, 100)
            }
            
            if self.gitlab_token:
                params['private_token'] = self.gitlab_token
            
            response = self.session.get(commits_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            for commit_data in data:
                commit = self._parse_gitlab_commit(commit_data)
                if commit:
                    commits.append(commit)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching GitLab repository commits: {e}")
        
        return commits
    
    def convert_to_public_source(self, commit: GitCommit) -> PublicSource:
        """Convert Git commit to public source format."""
        # Create content from commit information
        content = f"Commit: {commit.commit_hash}\n"
        content += f"Repository: {commit.repository}\n"
        content += f"Author: {commit.author} ({commit.author_email})\n"
        content += f"Date: {commit.commit_date.isoformat()}\n"
        content += f"Message: {commit.message}\n"
        
        if commit.files_changed:
            content += f"\nFiles changed:\n" + "\n".join(f"- {file}" for file in commit.files_changed)
        
        content += f"\nAdditions: {commit.additions}, Deletions: {commit.deletions}"
        
        return PublicSource(
            id=generate_id(f"commit_{commit.commit_hash}"),
            title=f"Git Commit: {commit.message[:100]}...",
            content=content,
            source_type=SourceType.GIT_COMMIT,
            url=f"https://github.com/{commit.repository}/commit/{commit.commit_hash}",
            published_date=commit.commit_date,
            author=commit.author,
            organization=commit.repository.split('/')[0] if '/' in commit.repository else None,
            metadata={
                "commit_hash": commit.commit_hash,
                "repository": commit.repository,
                "author_email": commit.author_email,
                "files_changed": commit.files_changed,
                "additions": commit.additions,
                "deletions": commit.deletions,
                "branch": commit.branch,
                "tags": commit.tags
            },
            tags=["git", "commit", "code"]
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


async def search_git_commits(query: str, max_results: int = 100, platforms: List[str] | None = None) -> List[PublicSource]:
    """Asynchronously search commits and return public source records."""
    # PLACEHOLDER endpoint. Returns no results until pointed at a real service
    # (e.g. the GitHub/GitLab search APIs defined in GitCommitsCrawler.apis).
    url = "https://example.com/commits"
    params = {"q": query, "n": max_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await _fetch_with_retry(client, url, params)
        data = response.json()
    except Exception:
        logger.exception("Commit search failed")
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
                title=item.get("message", ""),
                content=item.get("message", ""),
                source_type=SourceType.GIT_COMMIT,
                url=item.get("url"),
                published_date=published,
                author=item.get("author"),
                tags=["commit"],
            )
        )

    return results
