"""Blind prober for black-box red teaming.

Executes iterative probing campaigns: each round sends probe prompts derived
from the current hypothesis set, analyses the responses, and uses interesting
responses to refine hypotheses for the next round.
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..target_llm import ProbeResult, TargetLLMClient
from .hypothesis_engine import Hypothesis, HypothesisEngine
from .response_analyzer import AnomalySignal, ResponseAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BlackBoxRoundResult:
    """Results of a single probing round."""

    round_number: int
    hypotheses: List[Hypothesis]
    probe_results: List[ProbeResult]
    anomalies: List[AnomalySignal]
    flagged_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "hypotheses_count": len(self.hypotheses),
            "probe_results": [r.to_dict() for r in self.probe_results],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "flagged_count": self.flagged_count,
        }


class BlindProber:
    """Runs an iterative black-box probing campaign.

    Workflow per round:
    1. Convert current hypotheses into probe prompts.
    2. Send probes to the target LLM.
    3. Analyse responses for anomalous specificity.
    4. Use flagged responses to refine/extend hypotheses for next round.
    """

    def __init__(
        self,
        target: TargetLLMClient,
        hypothesis_engine: HypothesisEngine,
        analyzer: ResponseAnalyzer,
        max_rounds: int = 10,
        probes_per_hypothesis: int = 1,
        delay_seconds: float = 0.5,
    ):
        self.target = target
        self.hypothesis_engine = hypothesis_engine
        self.analyzer = analyzer
        self.max_rounds = max_rounds
        self.probes_per_hypothesis = probes_per_hypothesis
        self.delay_seconds = delay_seconds
        self.all_rounds: List[BlackBoxRoundResult] = []

    def run_campaign(
        self,
        initial_hypotheses: List[Hypothesis],
        output_dir: Optional[str] = None,
    ) -> List[BlackBoxRoundResult]:
        """Run the full iterative campaign.

        Args:
            initial_hypotheses: Starting hypotheses (from HypothesisEngine.generate).
            output_dir: If set, save round results as JSONL to this directory.
        """
        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        current_hypotheses = initial_hypotheses
        self.all_rounds = []

        for round_num in range(1, self.max_rounds + 1):
            if not current_hypotheses:
                logger.info(f"No hypotheses remaining at round {round_num}; stopping.")
                break

            logger.info(f"Black-box round {round_num}/{self.max_rounds}: {len(current_hypotheses)} hypotheses")
            round_result = self._run_round(round_num, current_hypotheses)
            self.all_rounds.append(round_result)

            if output_dir:
                self._save_round(round_result, output_dir)

            # Refine hypotheses from flagged responses
            refined = self._refine_hypotheses(round_result)
            # Avoid re-using queries already sent
            sent_queries = {h.query for h in current_hypotheses}
            current_hypotheses = [h for h in refined if h.query not in sent_queries]

            logger.info(
                f"Round {round_num} complete: "
                f"{round_result.flagged_count}/{len(round_result.probe_results)} flagged, "
                f"{len(current_hypotheses)} new hypotheses"
            )

        logger.info(f"Black-box campaign complete: {len(self.all_rounds)} rounds run.")
        return self.all_rounds

    def _run_round(self, round_num: int, hypotheses: List[Hypothesis]) -> BlackBoxRoundResult:
        """Execute a single round: probe each hypothesis, analyse responses."""
        probes_sent: List[ProbeResult] = []
        for hyp in hypotheses:
            # Send the hypothesis query as the probe directly
            result = self.target.send_probe(
                prompt=hyp.query,
                strategy=f"blackbox_round_{round_num}",
            )
            probes_sent.append(result)

        anomalies = self.analyzer.analyze_batch(probes_sent)
        flagged = [a for a in anomalies if a.flag]

        return BlackBoxRoundResult(
            round_number=round_num,
            hypotheses=hypotheses,
            probe_results=probes_sent,
            anomalies=anomalies,
            flagged_count=len(flagged),
        )

    def _refine_hypotheses(self, round_result: BlackBoxRoundResult) -> List[Hypothesis]:
        """Generate follow-up hypotheses from flagged responses."""
        new_hypotheses: List[Hypothesis] = []
        for probe_result, anomaly in zip(round_result.probe_results, round_result.anomalies):
            if not anomaly.flag:
                continue
            # Find the corresponding hypothesis
            hyp_query = probe_result.probe
            base_hyp = next(
                (h for h in round_result.hypotheses if h.query == hyp_query),
                Hypothesis(query=hyp_query, source="unknown"),
            )
            refined = self.hypothesis_engine.refine_from_response(
                hypothesis=base_hyp,
                response=probe_result.response,
                n=3,
            )
            new_hypotheses.extend(refined)
        return new_hypotheses

    def summarise(self) -> Dict[str, Any]:
        """Produce a campaign summary dict."""
        if not self.all_rounds:
            return {"total_rounds": 0, "total_probes": 0, "total_flagged": 0}

        total_probes = sum(len(r.probe_results) for r in self.all_rounds)
        total_flagged = sum(r.flagged_count for r in self.all_rounds)
        all_anomalies = [a for r in self.all_rounds for a in r.anomalies if a.flag]

        return {
            "total_rounds": len(self.all_rounds),
            "total_probes": total_probes,
            "total_flagged": total_flagged,
            "flag_rate": total_flagged / total_probes if total_probes else 0.0,
            "top_anomalies": [a.to_dict() for a in sorted(all_anomalies, key=lambda x: -x.specificity_score)[:5]],
        }

    @staticmethod
    def _save_round(round_result: BlackBoxRoundResult, output_dir: str) -> None:
        path = pathlib.Path(output_dir) / f"round_{round_result.round_number:03d}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(round_result.to_dict(), fh, indent=2)
