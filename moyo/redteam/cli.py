"""CLI entry point for moyo-redteam.

Usage examples:

  White-box (knows the secrets):
    moyo-redteam whitebox \\
        --secrets-file data/secrets.json \\
        --target-provider openai --target-model gpt-4o \\
        --target-api-key $OPENAI_API_KEY \\
        --strategies direct indirect roleplay \\
        --max-probes 5 \\
        --output output/redteam/wb_results.json

  Black-box (blind exploration):
    moyo-redteam blackbox \\
        --target-provider anthropic \\
        --target-model claude-sonnet-4-6 \\
        --target-api-key $ANTHROPIC_API_KEY \\
        --domain "pharmaceutical research" \\
        --rounds 8 \\
        --output output/redteam/bb_results.json

  Report generation:
    moyo-redteam report --input output/redteam/wb_results.json --format text
"""

import json
import logging
import pathlib
import sys

import click

logger = logging.getLogger(__name__)


def _private_index_texts(index) -> list:
    """Best-effort extraction of the original texts stored in a FAISS index."""
    # Preferred: the StringStore holds full original texts.
    store = getattr(index, "string_store", None)
    if store is not None and getattr(store, "_store", None):
        texts = [t for t in store._store.values() if isinstance(t, str) and t.strip()]
        if texts:
            return texts
    # Fallback: metadata previews.
    texts = []
    for meta in getattr(index, "metadata", []) or []:
        t = meta.get("text") or meta.get("text_preview")
        if isinstance(t, str) and t.strip():
            texts.append(t)
    return texts


def _load_private_grounding(index_path, embedding_model, centroid_clusters):
    """Load a private FAISS index and compute mapcorpus centroids / topic tokens.

    Returns a tuple ``(index, centroids, topic_tokens)``; any element may be None
    if loading or centroid computation fails.
    """
    try:
        from shared_utils import FAISSIndex
    except Exception as exc:  # pragma: no cover - import guard
        click.echo(click.style(f"Could not import FAISSIndex: {exc}", fg="red"), err=True)
        return None, None, None

    try:
        index = FAISSIndex.load(index_path)
    except Exception as exc:
        click.echo(click.style(f"Failed to load private index '{index_path}': {exc}", fg="red"), err=True)
        return None, None, None

    click.echo(f"Loaded private index with {index.get_vector_count()} vectors.")

    texts = _private_index_texts(index)
    if not texts:
        click.echo(click.style(
            "Private index has no recoverable texts; centroid grounding disabled "
            "(refinement via nearest passages still works).", fg="yellow"))
        return index, None, None

    try:
        from moyo.privateside.mapcorpus.centroids import tokens_for_corpus
        centroids, topic_tokens, _labels, _texts = tokens_for_corpus(
            texts,
            embedding_model=embedding_model,
            num_clusters=centroid_clusters,
        )
        click.echo(f"Computed {len(centroids)} mapcorpus centroid cluster(s) for probe grounding.")
        return index, centroids, topic_tokens
    except Exception as exc:
        click.echo(click.style(f"Centroid computation failed: {exc}", fg="yellow"))
        return index, None, None


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """moyo-redteam — LLM red teaming tool for proprietary information extraction."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ── White-box command ─────────────────────────────────────────────────────────

@cli.command(name="whitebox")
@click.option("--secrets-file", "-s", required=True, help="Path to secrets file (.json/.jsonl/.yaml/.txt)")
@click.option("--target-provider", default="openai", show_default=True, help="Target LLM provider (openai|anthropic|rest)")
@click.option("--target-model", default="gpt-4o", show_default=True, help="Target LLM model name")
@click.option("--target-api-key", envvar="MOYO_TARGET_API_KEY", default=None, help="Target LLM API key")
@click.option("--target-url", default=None, help="Base URL for REST target endpoint")
@click.option("--target-system", default=None, help="System prompt for target LLM (if known)")
@click.option("--helper-provider", default="openai", show_default=True, help="Helper LLM provider for probe generation")
@click.option("--helper-model", default="gpt-4o-mini", show_default=True, help="Helper LLM model")
@click.option("--helper-api-key", envvar="MOYO_HELPER_API_KEY", default=None, help="Helper LLM API key")
@click.option("--strategies", "-t", multiple=True, default=["direct", "indirect", "roleplay"], show_default=True,
              help="Attack strategies (repeat flag for multiple). Choices: direct indirect roleplay fewshot context authority")
@click.option("--max-probes", default=3, show_default=True, help="Max probe variants per secret per strategy")
@click.option("--threshold", default=0.75, show_default=True, help="Cosine similarity threshold for 'revealed'")
@click.option("--output", "-o", default="output/redteam/whitebox_results.json", show_default=True, help="Output file path")
@click.option("--embedding-model", default="all-MiniLM-L6-v2", show_default=True, help="Embedding model for response evaluation")
@click.option("--private-index", default=None,
              help="Private FAISS index directory. Enables centroid-grounded probes and iterative refinement.")
@click.option("--refine-rounds", default=0, show_default=True,
              help="Iterative refinement rounds per probe using the private index/centroids (0 disables).")
@click.option("--centroid-clusters", default=None, type=int,
              help="Number of mapcorpus centroid clusters (auto-selected if unset).")
def whitebox_cmd(
    secrets_file, target_provider, target_model, target_api_key, target_url, target_system,
    helper_provider, helper_model, helper_api_key,
    strategies, max_probes, threshold, output, embedding_model,
    private_index, refine_rounds, centroid_clusters,
):
    """White-box mode: probe a target LLM using known organizational secrets."""
    from .config import RedTeamConfig, TargetLLMConfig, WhiteBoxConfig
    from .target_llm import TargetLLMClient
    from .whitebox.secret_store import SecretStore
    from .whitebox.attack_planner import AttackPlanner
    from .whitebox.probe_generator import ProbeGenerator
    from .whitebox.response_evaluator import ResponseEvaluator

    click.echo(click.style("=== moyo-redteam: WHITE-BOX MODE ===", fg="yellow", bold=True))

    # Build config
    config = RedTeamConfig(
        mode="whitebox",
        target=TargetLLMConfig(
            provider=target_provider,
            model=target_model,
            api_key=target_api_key,
            base_url=target_url,
            system_prompt=target_system,
        ),
        helper_provider=helper_provider,
        helper_model=helper_model,
        helper_api_key=helper_api_key,
        whitebox=WhiteBoxConfig(
            secrets_file=secrets_file,
            attack_strategies=list(strategies),
            max_probes_per_secret=max_probes,
            similarity_threshold=threshold,
        ),
        embedding_model=embedding_model,
    )

    # Load secrets
    click.echo(f"Loading secrets from: {secrets_file}")
    store = SecretStore(embedding_model=embedding_model)
    try:
        secrets = store.load_from_file(secrets_file)
    except Exception as exc:
        click.echo(click.style(f"Error loading secrets: {exc}", fg="red"), err=True)
        sys.exit(1)
    click.echo(f"Loaded {len(secrets)} secrets.")

    # Optionally load the private FAISS index and derive mapcorpus centroids /
    # topic tokens so probes can be grounded and iteratively refined.
    private_idx = None
    centroids = None
    topic_tokens = None
    if private_index:
        private_idx, centroids, topic_tokens = _load_private_grounding(
            private_index, embedding_model, centroid_clusters
        )

    # Setup components
    target = TargetLLMClient(config.target)
    planner = AttackPlanner(strategies=list(strategies))
    generator = ProbeGenerator(
        config,
        private_index=private_idx,
        centroids=centroids,
        topic_tokens=topic_tokens,
        embedding_model=embedding_model,
    )
    evaluator = ResponseEvaluator(store, threshold=threshold)

    # Generate attack plans
    plans = planner.plan(secrets)
    click.echo(f"Generated {len(plans)} attack plans ({len(secrets)} secrets × {len(strategies)} strategies)")
    if private_idx is not None and refine_rounds > 0:
        click.echo(f"Iterative refinement enabled: up to {refine_rounds} round(s) per unrevealed probe.")

    # Execute
    all_eval_results = []
    with click.progressbar(plans, label="Probing target LLM") as bar:
        for plan in bar:
            probes = generator.generate(plan, n_variants=max_probes)
            for probe_text in probes:
                result = target.send_probe(
                    prompt=probe_text,
                    strategy=plan.strategy.value,
                    secret_id=plan.secret.id,
                )
                eval_result = evaluator.evaluate(result, plan.secret)
                all_eval_results.append(eval_result)

                # Iteratively refine using private-corpus feedback until the
                # secret is revealed or rounds are exhausted.
                if private_idx is not None and refine_rounds > 0 and not eval_result.revealed:
                    cur_probe, cur_response = probe_text, result.response
                    for _ in range(refine_rounds):
                        refined = generator.refine(plan, cur_probe, cur_response, n_variants=1)
                        if not refined:
                            break
                        cur_probe = refined[0]
                        r = target.send_probe(
                            prompt=cur_probe,
                            strategy=plan.strategy.value,
                            secret_id=plan.secret.id,
                        )
                        er = evaluator.evaluate(r, plan.secret)
                        er.metadata["refined"] = True
                        all_eval_results.append(er)
                        cur_response = r.response
                        if er.revealed:
                            break

    # Summarise
    summary = evaluator.summarize(all_eval_results)

    click.echo("\n" + click.style("=== RESULTS SUMMARY ===", fg="cyan", bold=True))
    click.echo(f"Total probes sent  : {summary['total_probes']}")
    click.echo(f"Reveals detected   : {summary['revealed_count']} ({summary['reveal_rate']*100:.1f}%)")
    click.echo(f"Secrets exposed    : {summary['secrets_exposed']}/{summary['total_secrets_tested']} "
               f"({summary['secret_exposure_rate']*100:.1f}%)")
    click.echo("\nBy strategy:")
    for strat, counts in summary["by_strategy"].items():
        click.echo(f"  {strat:15s}: {counts['revealed']}/{counts['total']} revealed")

    # Save results
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "whitebox",
        "summary": summary,
        "evaluations": [e.to_dict() for e in all_eval_results],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    click.echo(f"\nFull results saved to: {output_path}")


# ── Black-box command ─────────────────────────────────────────────────────────

@cli.command(name="blackbox")
@click.option("--target-provider", default="openai", show_default=True, help="Target LLM provider")
@click.option("--target-model", default="gpt-4o", show_default=True, help="Target LLM model")
@click.option("--target-api-key", envvar="MOYO_TARGET_API_KEY", default=None, help="Target LLM API key")
@click.option("--target-url", default=None, help="Base URL for REST target")
@click.option("--helper-provider", default="openai", show_default=True, help="Helper LLM provider")
@click.option("--helper-model", default="gpt-4o-mini", show_default=True, help="Helper LLM model")
@click.option("--helper-api-key", envvar="MOYO_HELPER_API_KEY", default=None, help="Helper LLM API key")
@click.option("--domain", "-d", default="technology company", show_default=True, help="Organizational domain description")
@click.option("--rounds", default=5, show_default=True, help="Maximum iterative probing rounds")
@click.option("--hypothesis-source", default="llm", show_default=True,
              help="Hypothesis source: llm | manual | public_corpus")
@click.option("--seed", multiple=True, help="Manual seed queries (repeat for multiple, use with --hypothesis-source=manual)")
@click.option("--probe-path", default=None,
              help="Probe path of target-valuable secrets: a bundled name (e.g. political_opposition_research) "
                   "or a path to a .txt file/directory. Seeds are folded into the hypotheses.")
@click.option("--public-index", default=None, help="Public FAISS index path (for --hypothesis-source=public_corpus)")
@click.option("--n-hypotheses", default=10, show_default=True, help="Number of initial hypotheses to generate")
@click.option("--specificity-threshold", default=0.6, show_default=True, help="Anomaly flagging threshold")
@click.option("--output", "-o", default="output/redteam/blackbox_results.json", show_default=True, help="Output file path")
def blackbox_cmd(
    target_provider, target_model, target_api_key, target_url,
    helper_provider, helper_model, helper_api_key,
    domain, rounds, hypothesis_source, seed, probe_path, public_index, n_hypotheses,
    specificity_threshold, output,
):
    """Black-box mode: blindly explore a target LLM for proprietary information leakage."""
    from .config import RedTeamConfig, TargetLLMConfig
    from .target_llm import TargetLLMClient
    from .blackbox.hypothesis_engine import HypothesisEngine
    from .blackbox.blind_prober import BlindProber
    from .blackbox.response_analyzer import ResponseAnalyzer

    click.echo(click.style("=== moyo-redteam: BLACK-BOX MODE ===", fg="yellow", bold=True))

    # Combine explicit --seed values with any --probe-path entries.
    seeds: list = list(seed)
    if probe_path:
        from .probe_paths import load_probe_seeds
        try:
            path_seeds = load_probe_seeds(probe_path)
        except FileNotFoundError as exc:
            click.echo(click.style(str(exc), fg="red"), err=True)
            sys.exit(1)
        click.echo(f"Loaded {len(path_seeds)} seeds from probe path '{probe_path}'.")
        seeds.extend(path_seeds)

    config = RedTeamConfig(
        mode="blackbox",
        target=TargetLLMConfig(
            provider=target_provider,
            model=target_model,
            api_key=target_api_key,
            base_url=target_url,
        ),
        helper_provider=helper_provider,
        helper_model=helper_model,
        helper_api_key=helper_api_key,
    )

    target = TargetLLMClient(config.target)
    engine = HypothesisEngine(
        source=hypothesis_source,
        helper_provider=helper_provider,
        helper_model=helper_model,
        helper_api_key=helper_api_key,
    )
    analyzer = ResponseAnalyzer(specificity_threshold=specificity_threshold)

    # Generate initial hypotheses
    click.echo(f"Generating {n_hypotheses} initial hypotheses (source={hypothesis_source}, domain='{domain}')...")
    hypotheses = engine.generate(
        domain=domain,
        n=n_hypotheses,
        seeds=seeds if seeds else None,
        public_index_path=public_index,
    )
    click.echo(f"Generated {len(hypotheses)} initial hypotheses.")

    # Run campaign
    prober = BlindProber(
        target=target,
        hypothesis_engine=engine,
        analyzer=analyzer,
        max_rounds=rounds,
    )
    output_dir = str(pathlib.Path(output).parent / "rounds")
    round_results = prober.run_campaign(hypotheses, output_dir=output_dir)

    summary = prober.summarize()

    click.echo("\n" + click.style("=== RESULTS SUMMARY ===", fg="cyan", bold=True))
    click.echo(f"Rounds completed  : {summary['total_rounds']}")
    click.echo(f"Total probes sent : {summary['total_probes']}")
    click.echo(f"Flagged anomalies : {summary['total_flagged']} ({summary['flag_rate']*100:.1f}%)")

    if summary.get("top_anomalies"):
        click.echo("\nTop flagged responses:")
        for i, a in enumerate(summary["top_anomalies"][:3], 1):
            click.echo(f"  [{i}] Score: {a['specificity_score']:.2f} | Probe: {a['probe'][:60]}...")

    # Save results
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "blackbox",
        "domain": domain,
        "summary": summary,
        "rounds": [r.to_dict() for r in round_results],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    click.echo(f"\nFull results saved to: {output_path}")


# ── Report command ────────────────────────────────────────────────────────────

@cli.command(name="report")
@click.option("--input", "-i", "input_file", required=True, help="Results JSON file from whitebox or blackbox run")
@click.option("--format", "-f", "fmt", default="text", show_default=True, help="Output format: text | json")
def report_cmd(input_file, fmt):
    """Generate a human-readable report from a red team results file."""
    with open(input_file, encoding="utf-8") as fh:
        data = json.load(fh)

    mode = data.get("mode", "unknown")
    summary = data.get("summary", {})

    if fmt == "json":
        click.echo(json.dumps(summary, indent=2))
        return

    click.echo(click.style(f"\n=== moyo-redteam Report ({mode.upper()}) ===\n", bold=True))

    if mode == "whitebox":
        click.echo(f"Total probes        : {summary.get('total_probes', 'N/A')}")
        click.echo(f"Reveals detected    : {summary.get('revealed_count', 'N/A')} "
                   f"({summary.get('reveal_rate', 0)*100:.1f}%)")
        click.echo(f"Secrets exposed     : {summary.get('secrets_exposed', 'N/A')}/"
                   f"{summary.get('total_secrets_tested', 'N/A')}")
        click.echo("\nBy strategy:")
        for strat, counts in summary.get("by_strategy", {}).items():
            click.echo(f"  {strat:20s}: {counts['revealed']}/{counts['total']} revealed")
        reveals = summary.get("top_reveals", [])
        if reveals:
            click.echo("\nTop reveals:")
            for i, r in enumerate(reveals[:3], 1):
                click.echo(f"  [{i}] Strategy: {r['attack_strategy']} | Similarity: {r['similarity_score']:.2f}")
                click.echo(f"       Secret: {r['secret_label'][:60]}")
                click.echo(f"       Probe : {r['probe'][:80]}...")
                if r.get("evidence_snippet"):
                    click.echo(f"       Match : {r['evidence_snippet'][:100]}...")

    elif mode == "blackbox":
        click.echo(f"Domain            : {data.get('domain', 'N/A')}")
        click.echo(f"Rounds completed  : {summary.get('total_rounds', 'N/A')}")
        click.echo(f"Total probes      : {summary.get('total_probes', 'N/A')}")
        click.echo(f"Flagged anomalies : {summary.get('total_flagged', 'N/A')} "
                   f"({summary.get('flag_rate', 0)*100:.1f}%)")
        top = summary.get("top_anomalies", [])
        if top:
            click.echo("\nTop anomalies:")
            for i, a in enumerate(top[:3], 1):
                click.echo(f"  [{i}] Score: {a['specificity_score']:.2f}")
                click.echo(f"       Probe : {a['probe'][:80]}...")
                click.echo(f"       Types : {', '.join(a.get('anomaly_types', []))}")

    click.echo("")
