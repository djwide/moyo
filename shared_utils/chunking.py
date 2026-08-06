"""Shared text chunking utilities for sente and moyo projects."""

import re
from dataclasses import dataclass
from typing import List, Iterator, Optional, Callable

# A markdown heading line, e.g. "## Section 2". Up to 3 leading spaces allowed.
_HEADING_RE = re.compile(r'^\s{0,3}#{1,6}\s')
# A horizontal rule on its own line, e.g. "---", "***", "___".
_HR_RE = re.compile(r'^\s*([-*_])\1{2,}\s*$')
# Sentence boundary: end punctuation followed by whitespace, but not inside a
# decimal number (e.g. "3.14"). The punctuation stays attached to the sentence.
_SENTENCE_SPLIT_RE = re.compile(r'(?<!\d)(?<=[.!?])\s+(?!\d)')
# Bullet / numbered / checkbox list items. Captures the idea text after the marker.
_LIST_ITEM_RE = re.compile(
    r'^\s*(?:'
    r'[-*+•▪◦]|'          # bullets
    r'[□☐☑✓✗]|'           # checkboxes
    r'\d+[.)]|'           # numbered: 1. or 1)
    r'[A-Za-z][.)]|'      # lettered: a. or a)
    r'\(\d+\)'            # parenthesized: (1)
    r')\s+(?P<item>.+?)\s*$'
)
# A short label line that introduces the next idea, e.g. "Purpose:" / "Notes:".
_LABEL_ONLY_RE = re.compile(r'^[A-Za-z][\w\s/&-]{0,40}:\s*$')
# Markdown bold key/value on one line, e.g. "**Status:** DO NOT DISTRIBUTE".
_KEY_VALUE_RE = re.compile(r'^\s*\*\*[^*]+\*\*\s*:?\s+\S')


def estimate_token_count(text: str) -> int:
    """Rough token estimate without loading a tokenizer.

    Uses ~4/3 tokens per whitespace-delimited word, which slightly over-counts
    for English prose so it acts as a safe upper bound when capping chunks
    against an embedding model's token limit.
    """
    words = re.findall(r"\S+", text)
    if not words:
        return 0
    return int(len(words) * 4 / 3) + 1


def chunk_text(text: str,
               chunk_size: int = 512,
               overlap: int = 50,
               preserve_sentences: bool = True,
               preserve_structure: bool = True,
               max_tokens: int = 256,
               tokenizer=None) -> List[str]:
    """Split text into structure-aware, token-bounded chunks.

    Chunks respect document structure (markdown headings and horizontal rules)
    and sentence boundaries, and are capped by both a character budget and a
    token limit so they never overflow the embedding model. Overlap is carried
    as whole trailing sentences (never mid-word character slices).

    Args:
        text: Input text to chunk
        chunk_size: Soft maximum size of each chunk in characters
        overlap: Approximate characters of trailing context to repeat between
            chunks, snapped to sentence (or word) boundaries. 0 disables it.
        preserve_sentences: Keep sentences intact when packing chunks
        preserve_structure: Split on markdown headings / horizontal rules first
            so a chunk stays within one section (heading kept with its content)
        max_tokens: Hard ceiling on tokens per chunk. Defaults to 256, the max
            sequence length of all-MiniLM-L6-v2; raise it for larger models.
        tokenizer: Optional tokenizer with an ``encode`` method for exact token
            counts. Falls back to :func:`estimate_token_count` when omitted.

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    if not preserve_sentences:
        return chunk_text_simple(text, chunk_size, overlap)

    if tokenizer is not None:
        count_tokens: Callable[[str], int] = lambda s: len(tokenizer.encode(s))
    else:
        count_tokens = estimate_token_count

    if preserve_structure:
        blocks = _split_structural_blocks(text)
    else:
        blocks = [re.sub(r'\s+', ' ', text.strip())]

    chunks: List[str] = []
    for block in blocks:
        chunks.extend(_pack_block(block, chunk_size, overlap, max_tokens, count_tokens))

    if not chunks:
        # Nothing was packable (e.g. no sentence boundaries); fall back.
        return chunk_text_simple(re.sub(r'\s+', ' ', text.strip()), chunk_size, overlap)

    return chunks


@dataclass
class GranularChunk:
    """A chunk emitted at a specific granularity level.

    Attributes:
        text: The chunk text.
        level: Granularity level:
            - ``"section"``: variable-size structure-aware chunk
            - ``"sentence"``: a sentence within a section
            - ``"item"``: a list item, bullet, checkbox, or other atomic line idea
        index: Position of this chunk in the emitted sequence for the document.
        parent_index: For finer chunks, the ``index`` of the section chunk they
            came from; ``None`` for section-level chunks.
    """
    text: str
    level: str
    index: int
    parent_index: Optional[int] = None


def chunk_text_multi_granularity(text: str,
                                 chunk_size: int = 512,
                                 overlap: int = 50,
                                 max_tokens: int = 256,
                                 include_sentences: bool = True,
                                 include_items: bool = True,
                                 min_sentence_chars: int = 15,
                                 min_item_chars: int = 3,
                                 tokenizer=None) -> List[GranularChunk]:
    """Chunk text into variable sizes at multiple granularities.

    Emits, in document order:

    1. *section* chunks from :func:`chunk_text` (structure-aware, token-bounded)
    2. *sentence* sub-chunks within each multi-sentence section (optional)
    3. *item* sub-chunks for atomic idea units that would otherwise be diluted
       inside a section: bullet points, numbered lists, checkboxes, key/value
       metadata lines, and other short standalone lines (optional)

    Items are extracted from the original line structure *before* whitespace is
    collapsed, so ``1. Bright citrus opening`` and ``* Lemon oil`` become their
    own vectors. Each fine chunk links back to its parent section via
    ``parent_index``.

    Args:
        text: Input text to chunk.
        chunk_size: Soft maximum characters per section chunk.
        overlap: Approximate characters of sentence-boundary overlap between
            consecutive section chunks.
        max_tokens: Hard token ceiling per section chunk (see :func:`chunk_text`).
        include_sentences: Also emit sentence-level sub-chunks.
        include_items: Also emit list-item / atomic-line sub-chunks.
        min_sentence_chars: Drop sentence sub-chunks shorter than this.
        min_item_chars: Drop item sub-chunks shorter than this.
        tokenizer: Optional tokenizer for exact token counts.

    Returns:
        List of :class:`GranularChunk` in document order.
    """
    sections = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        preserve_sentences=True,
        preserve_structure=True,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )

    chunks: List[GranularChunk] = []
    # Normalized section text → section index, for parent linking of items.
    section_lookup: List[tuple] = []  # (normalized_text, section_index)

    for section in sections:
        section_index = len(chunks)
        chunks.append(GranularChunk(text=section, level="section", index=section_index))
        section_lookup.append((_normalize_ws(section).lower(), section_index))

        if not include_sentences:
            continue

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(section) if s.strip()]
        # Only add sentence-level chunks when they actually subdivide the
        # section; a single-sentence section would just duplicate the vector.
        if len(sentences) <= 1:
            continue
        for sentence in sentences:
            if len(sentence) < min_sentence_chars:
                continue
            # Skip sentences that are really list items already captured as
            # items (e.g. "Bright citrus opening 2." from a mangled split).
            if _looks_like_list_residue(sentence):
                continue
            chunks.append(GranularChunk(
                text=sentence,
                level="sentence",
                index=len(chunks),
                parent_index=section_index,
            ))

    if include_items:
        seen = {_normalize_ws(c.text).lower() for c in chunks}
        for item_text in _extract_atomic_items(text, min_item_chars):
            key = _normalize_ws(item_text).lower()
            if key in seen:
                continue
            seen.add(key)
            parent_index = _find_parent_section(key, section_lookup)
            chunks.append(GranularChunk(
                text=item_text,
                level="item",
                index=len(chunks),
                parent_index=parent_index,
            ))

    return chunks


def _normalize_ws(text: str) -> str:
    """Collapse whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def _looks_like_list_residue(text: str) -> bool:
    """True if a sentence looks like a mangled list-item fragment.

    E.g. ``"Bright citrus opening 2."`` left over after sentence-splitting a
    numbered list that was whitespace-collapsed.
    """
    return bool(re.search(r'\s\d+\.\s*$', text.strip()))


def _extract_atomic_items(text: str, min_chars: int) -> List[str]:
    """Pull atomic idea units from the original line-oriented text.

    Captures:
    - Bullet / numbered / checkbox list items (marker stripped)
    - Markdown key/value metadata lines (``**Status:** DO NOT DISTRIBUTE``)
    - Short standalone non-heading lines that sit alone as their own idea
      (e.g. a one-line purpose statement after ``Purpose:``)

    Prose paragraphs spanning multiple lines are left for sentence splitting.
    """
    items: List[str] = []
    # Buffer consecutive non-list prose lines so we can decide whether the
    # block is a short standalone idea or multi-line prose.
    prose_buf: List[str] = []

    def flush_prose() -> None:
        if not prose_buf:
            return
        # Drop trailing label-only lines ("Purpose:") — they introduce content
        # rather than carrying an idea themselves.
        while prose_buf and _LABEL_ONLY_RE.match(prose_buf[-1]):
            prose_buf.pop()
        # Leading label-only lines are context; keep only the body after them.
        body_start = 0
        while body_start < len(prose_buf) and _LABEL_ONLY_RE.match(prose_buf[body_start]):
            body_start += 1
        body = prose_buf[body_start:]
        prose_buf.clear()
        if not body:
            return
        # A short single-line (or two-line) idea that isn't a full paragraph of
        # sentences becomes its own item vector.
        joined = _normalize_ws(' '.join(body))
        if not joined or len(joined) < min_chars:
            return
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(joined) if s.strip()]
        if len(body) <= 2 and len(sentences) <= 1 and len(joined) <= 200:
            items.append(joined)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped or _HR_RE.match(stripped) or _HEADING_RE.match(stripped):
            flush_prose()
            continue

        list_match = _LIST_ITEM_RE.match(stripped)
        if list_match:
            flush_prose()
            item = list_match.group('item').strip()
            # Strip trailing markdown emphasis leftovers if any.
            item = item.strip('*_ ')
            if len(item) >= min_chars:
                items.append(item)
            continue

        if _KEY_VALUE_RE.match(stripped):
            flush_prose()
            # Prefer the value side when the line is "**Key:** value".
            value = re.sub(r'^\s*\*\*[^*]+\*\*\s*:?\s*', '', stripped).strip()
            if len(value) >= min_chars:
                items.append(value)
            elif len(stripped) >= min_chars:
                items.append(_normalize_ws(stripped))
            continue

        if _LABEL_ONLY_RE.match(stripped):
            # Keep in the prose buffer so a following short body can be paired,
            # but don't emit the label alone.
            prose_buf.append(stripped)
            continue

        prose_buf.append(stripped)

    flush_prose()
    return items


def _find_parent_section(normalized_item: str, section_lookup: List[tuple]) -> Optional[int]:
    """Return the section index whose text contains the item, if any."""
    for section_text, section_index in section_lookup:
        if normalized_item in section_text:
            return section_index
    return None


def _split_structural_blocks(text: str) -> List[str]:
    """Segment text into section blocks on headings and horizontal rules.

    A heading starts a new block and is kept together with the content that
    follows it (so the heading provides context to its section's chunks).
    Horizontal rules act as hard separators and are dropped. Whitespace within
    each block is normalized to single spaces.
    """
    blocks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            joined = re.sub(r'\s+', ' ', ' '.join(current)).strip()
            if joined:
                blocks.append(joined)
            current.clear()

    def has_body() -> bool:
        return any(
            existing.strip() and not _HEADING_RE.match(existing) for existing in current
        )

    for line in text.splitlines():
        if not line.strip():
            # Blank lines only separate paragraphs; whitespace is normalized
            # per block anyway, so they carry no structural weight here.
            continue
        if _HR_RE.match(line):
            flush()
            continue
        if _HEADING_RE.match(line):
            # Only start a new block once the current one has body text, so
            # stacked headings (title + subtitle) stay with their content.
            if has_body():
                flush()
            current.append(line.strip())
            continue
        current.append(line)

    flush()

    if not blocks:
        normalized = re.sub(r'\s+', ' ', text.strip())
        return [normalized] if normalized else []
    return blocks


def _pack_block(block: str,
                chunk_size: int,
                overlap: int,
                max_tokens: int,
                count_tokens: Callable[[str], int]) -> List[str]:
    """Pack a single block's sentences into char- and token-bounded chunks."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(block) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []

    def current_text() -> str:
        return " ".join(current)

    def exceeds_limits(candidate: str) -> bool:
        if len(candidate) > chunk_size:
            return True
        if max_tokens and count_tokens(candidate) > max_tokens:
            return True
        return False

    for sentence in sentences:
        # A single sentence that busts the limits gets hard-split on words.
        if exceeds_limits(sentence):
            if current:
                chunks.append(current_text())
                current = []
            chunks.extend(_hard_split(sentence, chunk_size, overlap, max_tokens, count_tokens))
            continue

        candidate = (current_text() + " " + sentence).strip() if current else sentence
        if current and exceeds_limits(candidate):
            chunks.append(current_text())
            current = _overlap_tail(current, overlap)
        current.append(sentence)

    if current:
        chunks.append(current_text())

    return chunks


def _overlap_tail(units: List[str], overlap: int) -> List[str]:
    """Return trailing units (sentences/words) totalling up to ``overlap`` chars."""
    if overlap <= 0 or not units:
        return []
    carried: List[str] = []
    total = 0
    for unit in reversed(units):
        addition = len(unit) + (1 if carried else 0)
        if carried and total + addition > overlap:
            break
        carried.insert(0, unit)
        total += addition
    return carried


def _hard_split(text: str,
                chunk_size: int,
                overlap: int,
                max_tokens: int,
                count_tokens: Callable[[str], int]) -> List[str]:
    """Split an oversized sentence on word boundaries (never mid-word)."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current: List[str] = []

    def current_text() -> str:
        return " ".join(current)

    for word in words:
        candidate = (current_text() + " " + word).strip() if current else word
        over = len(candidate) > chunk_size or (max_tokens and count_tokens(candidate) > max_tokens)
        if current and over:
            chunks.append(current_text())
            current = _overlap_tail(current, overlap)
        current.append(word)

    if current:
        chunks.append(current_text())

    return chunks


def chunk_text_simple(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple character-based chunking with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_lines(lines: Iterator[str], 
                max_lines_per_chunk: int = 10,
                max_chunk_size: Optional[int] = None) -> Iterator[str]:
    """Chunk text by grouping lines together.
    
    Args:
        lines: Iterator of text lines
        max_lines_per_chunk: Maximum number of lines per chunk
        max_chunk_size: Maximum character size per chunk (optional)
        
    Yields:
        Text chunks as strings
    """
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_size = len(line)
        
        # Check if adding this line would exceed limits
        if (len(current_chunk) >= max_lines_per_chunk or 
            (max_chunk_size and current_size + line_size > max_chunk_size)):
            if current_chunk:
                yield "\n".join(current_chunk)
                current_chunk = []
                current_size = 0
        
        current_chunk.append(line)
        current_size += line_size
    
    # Yield remaining lines
    if current_chunk:
        yield "\n".join(current_chunk)


def chunk_by_tokens(text: str, 
                   tokenizer=None,
                   max_tokens: int = 512,
                   overlap_tokens: int = 50) -> List[str]:
    """Chunk text by tokens using a tokenizer.
    
    Args:
        text: Input text to chunk
        tokenizer: Tokenizer to use (e.g., from transformers)
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap
        
    Returns:
        List of text chunks
    """
    if not tokenizer:
        # Fallback to character-based chunking
        return chunk_text(text, max_tokens * 4, overlap_tokens * 4)
    
    try:
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    except Exception:
        # Fallback to character-based chunking
        return chunk_text(text, max_tokens * 4, overlap_tokens * 4)
