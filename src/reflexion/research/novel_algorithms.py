"""Novel reflexion algorithms and comparative research implementation."""

import asyncio
# Use native Python instead of numpy for broader compatibility
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback implementations for numpy functions
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def var(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        
        @staticmethod
        def percentile(values, p):
            if not values:
                return 0
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * p / 100
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_values):
                return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
            else:
                return sorted_values[f]
        
        @staticmethod
        def argmax(values):
            if not values:
                return 0
            return max(range(len(values)), key=lambda i: values[i])
        
        @staticmethod
        def argmin(values):
            if not values:
                return 0
            return min(range(len(values)), key=lambda i: values[i])
        
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        # Add ndarray placeholder
        class ndarray:
            pass
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import random
import math

from ..core.types import ReflexionResult, Reflection
from ..core.logging_config import logger


class ReflexionAlgorithm(Enum):
    """Available reflexion algorithms for research comparison."""
    CLASSIC_REFLEXION = "classic_reflexion"
    HIERARCHICAL_REFLEXION = "hierarchical_reflexion"
    ENSEMBLE_REFLEXION = "ensemble_reflexion"
    ADAPTIVE_REFLEXION = "adaptive_reflexion"
    QUANTUM_INSPIRED_REFLEXION = "quantum_inspired_reflexion"
    EVOLUTIONARY_REFLEXION = "evolutionary_reflexion"
    META_COGNITIVE_REFLEXION = "meta_cognitive_reflexion"
    CONTRASTIVE_REFLEXION = "contrastive_reflexion"
    MULTI_MODAL_REFLEXION = "multi_modal_reflexion"
    TEMPORAL_REFLEXION = "temporal_reflexion"


@dataclass
class AlgorithmPerformance:
    """Performance metrics for reflexion algorithms."""
    algorithm: ReflexionAlgorithm
    success_rate: float
    avg_iterations: float
    avg_execution_time: float
    avg_reflection_quality: float
    convergence_rate: float
    stability_score: float
    adaptability_score: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    statistical_significance: float


@dataclass
class ReflexionState:
    """Enhanced state representation for advanced algorithms."""
    iteration: int
    task: str
    current_output: str
    historical_outputs: List[str]
    success_scores: List[float]
    reflections: List[Reflection]
    meta_reflections: List[Dict[str, Any]]
    context_embeddings: Optional[Any] = None
    confidence_trajectory: List[float] = field(default_factory=list)
    improvement_trajectory: List[float] = field(default_factory=list)
    strategy_history: List[str] = field(default_factory=list)


class HierarchicalReflexionAlgorithm:
    """Multi-level hierarchical reflexion with strategic decomposition."""
    
    def __init__(self, levels: int = 3, focus_threshold: float = 0.3):
        self.levels = levels
        self.focus_threshold = focus_threshold
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, state: ReflexionState, llm_provider) -> Tuple[bool, ReflexionState]:
        """Execute hierarchical reflexion algorithm."""
        
        # Level 1: Strategic decomposition
        strategic_reflection = await self._generate_strategic_reflection(state, llm_provider)
        
        # Level 2: Tactical analysis
        tactical_reflections = await self._generate_tactical_reflections(
            state, strategic_reflection, llm_provider
        )
        
        # Level 3: Operational improvements
        operational_improvements = await self._generate_operational_improvements(
            state, tactical_reflections, llm_provider
        )
        
        # Synthesize multi-level insights
        synthesized_reflection = self._synthesize_hierarchical_insights(
            strategic_reflection, tactical_reflections, operational_improvements
        )
        
        # Apply improvements hierarchically
        improved_state = await self._apply_hierarchical_improvements(
            state, synthesized_reflection, llm_provider
        )
        
        # Evaluate success with hierarchical criteria
        success = self._evaluate_hierarchical_success(improved_state)
        
        return success, improved_state
    
    async def _generate_strategic_reflection(self, state: ReflexionState, llm_provider) -> Dict[str, Any]:
        """Generate high-level strategic reflection."""
        strategic_prompt = f"""
        Analyze the overall strategy and approach for this task at a high level:
        
        Task: {state.task}
        Current Iteration: {state.iteration}
        Historical Performance: {state.success_scores[-3:] if state.success_scores else []}
        
        Focus on:
        1. Is the fundamental approach correct?
        2. Are we solving the right problem?
        3. What strategic pivots might be needed?
        4. How does this task relate to broader objectives?
        
        Provide strategic insights in JSON format.
        """
        
        response = await llm_provider.generate_async(strategic_prompt)
        
        try:
            strategic_data = json.loads(response)
        except json.JSONDecodeError:
            strategic_data = {
                "fundamental_approach": "needs_analysis",
                "problem_alignment": "uncertain",
                "strategic_pivots": ["refactor_approach"],
                "broader_context": "task_specific"
            }
        
        return strategic_data
    
    async def _generate_tactical_reflections(
        self, state: ReflexionState, strategic_reflection: Dict[str, Any], llm_provider
    ) -> List[Dict[str, Any]]:
        """Generate mid-level tactical reflections."""
        
        tactical_areas = [
            "execution_methodology",
            "resource_allocation", 
            "quality_assurance",
            "risk_mitigation"
        ]
        
        tactical_reflections = []
        
        for area in tactical_areas:
            tactical_prompt = f"""
            Analyze the tactical approach for {area} given the strategic context:
            
            Strategic Context: {strategic_reflection}
            Current Execution: {state.current_output[:500]}...
            
            Focus specifically on {area}:
            - Current effectiveness
            - Improvement opportunities
            - Resource optimization
            - Quality enhancement
            
            Provide tactical analysis in JSON format.
            """
            
            response = await llm_provider.generate_async(tactical_prompt)
            
            try:
                tactical_data = json.loads(response)
                tactical_data["area"] = area
                tactical_reflections.append(tactical_data)
            except json.JSONDecodeError:
                tactical_reflections.append({
                    "area": area,
                    "effectiveness": "moderate",
                    "improvements": ["optimize_execution"],
                    "resources": ["time_management"],
                    "quality": ["add_validation"]
                })
        
        return tactical_reflections
    
    async def _generate_operational_improvements(
        self, state: ReflexionState, tactical_reflections: List[Dict[str, Any]], llm_provider
    ) -> List[Dict[str, Any]]:
        """Generate specific operational improvements."""
        
        operational_improvements = []
        
        for tactical in tactical_reflections:
            operational_prompt = f"""
            Generate specific operational improvements for {tactical['area']}:
            
            Tactical Context: {tactical}
            Current Output: {state.current_output}
            Recent Reflections: {[r.improvements for r in state.reflections[-2:]] if state.reflections else []}
            
            Provide concrete, actionable operational improvements:
            - Specific changes to make
            - Implementation steps
            - Expected impact
            - Success metrics
            
            Respond in JSON format.
            """
            
            response = await llm_provider.generate_async(operational_prompt)
            
            try:
                operational_data = json.loads(response)
                operational_data["tactical_area"] = tactical["area"]
                operational_improvements.append(operational_data)
            except json.JSONDecodeError:
                operational_improvements.append({
                    "tactical_area": tactical["area"],
                    "specific_changes": ["improve_implementation"],
                    "steps": ["analyze", "implement", "validate"],
                    "expected_impact": "moderate_improvement",
                    "success_metrics": ["quality_score", "execution_time"]
                })
        
        return operational_improvements
    
    def _synthesize_hierarchical_insights(
        self,
        strategic: Dict[str, Any],
        tactical: List[Dict[str, Any]],
        operational: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize insights across all hierarchical levels."""
        
        return {
            "strategic_insights": strategic,
            "tactical_insights": tactical,
            "operational_improvements": operational,
            "synthesis": {
                "priority_actions": self._extract_priority_actions(strategic, tactical, operational),
                "coherence_score": self._calculate_coherence_score(strategic, tactical, operational),
                "implementation_plan": self._generate_implementation_plan(operational),
                "success_predictors": self._identify_success_predictors(strategic, tactical)
            }
        }
    
    def _extract_priority_actions(
        self, strategic: Dict, tactical: List[Dict], operational: List[Dict]
    ) -> List[str]:
        """Extract highest priority actions across all levels."""
        priority_actions = []
        
        # Strategic priorities
        if "strategic_pivots" in strategic:
            priority_actions.extend(strategic["strategic_pivots"][:2])
        
        # Tactical priorities
        for t in tactical:
            if "improvements" in t:
                priority_actions.extend(t["improvements"][:1])
        
        # Operational priorities
        for o in operational:
            if "specific_changes" in o:
                priority_actions.extend(o["specific_changes"][:1])
        
        # Remove duplicates and prioritize
        unique_actions = list(dict.fromkeys(priority_actions))
        return unique_actions[:5]  # Top 5 priorities
    
    def _calculate_coherence_score(
        self, strategic: Dict, tactical: List[Dict], operational: List[Dict]
    ) -> float:
        """Calculate coherence score across hierarchical levels."""
        # Simplified coherence calculation
        # In practice, this would use semantic similarity
        coherence_factors = []
        
        # Strategic-tactical alignment
        strategic_themes = set(str(strategic).lower().split())
        tactical_themes = set()
        for t in tactical:
            tactical_themes.update(str(t).lower().split())
        
        strategic_tactical_overlap = len(strategic_themes & tactical_themes) / max(len(strategic_themes), 1)
        coherence_factors.append(strategic_tactical_overlap)
        
        # Tactical-operational alignment
        operational_themes = set()
        for o in operational:
            operational_themes.update(str(o).lower().split())
        
        tactical_operational_overlap = len(tactical_themes & operational_themes) / max(len(tactical_themes), 1)
        coherence_factors.append(tactical_operational_overlap)
        
        return min(1.0, np.mean(coherence_factors) * 2)  # Scale to 0-1
    
    def _generate_implementation_plan(self, operational: List[Dict]) -> List[Dict[str, Any]]:
        """Generate sequenced implementation plan."""
        plan = []
        
        for i, op in enumerate(operational):
            if "steps" in op and "expected_impact" in op:
                plan.append({
                    "phase": i + 1,
                    "area": op.get("tactical_area", f"area_{i}"),
                    "steps": op["steps"],
                    "expected_impact": op["expected_impact"],
                    "dependencies": [] if i == 0 else [f"phase_{i}"]
                })
        
        return plan
    
    def _identify_success_predictors(self, strategic: Dict, tactical: List[Dict]) -> List[str]:
        """Identify key success predictors."""
        predictors = []
        
        # From strategic level
        if "broader_context" in strategic:
            predictors.append(f"strategic_alignment_{strategic['broader_context']}")
        
        # From tactical level
        for t in tactical:
            if "quality" in t:
                predictors.extend([f"tactical_quality_{t['area']}" for _ in t["quality"][:1]])
        
        return predictors[:3]  # Top 3 predictors
    
    async def _apply_hierarchical_improvements(
        self, state: ReflexionState, synthesized: Dict[str, Any], llm_provider
    ) -> ReflexionState:
        """Apply improvements based on hierarchical analysis."""
        
        improvement_prompt = f"""
        Apply the following hierarchical improvements to enhance the current output:
        
        Current Output: {state.current_output}
        Priority Actions: {synthesized['synthesis']['priority_actions']}
        Implementation Plan: {synthesized['synthesis']['implementation_plan']}
        
        Provide an improved version that incorporates:
        1. Strategic alignment improvements
        2. Tactical execution enhancements  
        3. Operational optimizations
        
        Focus on coherent integration across all levels.
        """
        
        improved_output = await llm_provider.generate_async(improvement_prompt)
        
        # Update state
        new_state = ReflexionState(
            iteration=state.iteration + 1,
            task=state.task,
            current_output=improved_output,
            historical_outputs=state.historical_outputs + [state.current_output],
            success_scores=state.success_scores,
            reflections=state.reflections,
            meta_reflections=state.meta_reflections + [synthesized]
        )
        
        return new_state
    
    def _evaluate_hierarchical_success(self, state: ReflexionState) -> bool:
        """Evaluate success using hierarchical criteria."""
        
        # Multiple success criteria across levels
        criteria_met = 0
        total_criteria = 3
        
        # Strategic success: coherent approach
        if state.meta_reflections:
            latest_meta = state.meta_reflections[-1]
            if latest_meta.get("synthesis", {}).get("coherence_score", 0) > 0.7:
                criteria_met += 1
        
        # Tactical success: quality improvement
        if len(state.historical_outputs) > 1:
            # Simplified improvement detection
            if len(state.current_output) > len(state.historical_outputs[-1]):
                criteria_met += 1
        
        # Operational success: actionable output
        if len(state.current_output) > 100:  # Substantial output
            criteria_met += 1
        
        return criteria_met >= 2  # At least 2/3 criteria


class EnsembleReflexionAlgorithm:
    """Ensemble approach combining multiple reflexion strategies."""
    
    def __init__(self, strategies: List[str] = None, voting_threshold: float = 0.6):
        self.strategies = strategies or [
            "analytical", "creative", "systematic", "intuitive", "critical"
        ]
        self.voting_threshold = voting_threshold
        self.strategy_weights = {s: 1.0 for s in self.strategies}
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, state: ReflexionState, llm_provider) -> Tuple[bool, ReflexionState]:
        """Execute ensemble reflexion with multiple strategies."""
        
        # Generate reflections from each strategy
        strategy_reflections = await self._generate_strategy_reflections(state, llm_provider)
        
        # Evaluate and weight strategy outputs
        weighted_reflections = self._weight_strategy_reflections(strategy_reflections, state)
        
        # Ensemble voting on improvements
        consensus_improvements = self._ensemble_voting(weighted_reflections)
        
        # Generate ensemble-informed output
        improved_state = await self._apply_ensemble_improvements(
            state, consensus_improvements, llm_provider
        )
        
        # Adaptive weight adjustment
        self._adjust_strategy_weights(strategy_reflections, improved_state)
        
        # Evaluate ensemble success
        success = self._evaluate_ensemble_success(improved_state, consensus_improvements)
        
        return success, improved_state
    
    async def _generate_strategy_reflections(
        self, state: ReflexionState, llm_provider
    ) -> Dict[str, Dict[str, Any]]:
        """Generate reflections from different strategic perspectives."""
        
        strategy_prompts = {
            "analytical": f"""
            Analyze the current output analytically and systematically:
            Task: {state.task}
            Output: {state.current_output}
            
            Focus on:
            - Logical structure and reasoning
            - Data accuracy and evidence
            - Methodological rigor
            - Quantitative improvements
            
            Provide analytical reflection in JSON format.
            """,
            
            "creative": f"""
            Approach the current output from a creative and innovative perspective:
            Task: {state.task}
            Output: {state.current_output}
            
            Focus on:
            - Novel approaches and alternatives
            - Creative problem-solving
            - Unconventional solutions
            - Imaginative enhancements
            
            Provide creative reflection in JSON format.
            """,
            
            "systematic": f"""
            Evaluate the current output systematically and comprehensively:
            Task: {state.task}
            Output: {state.current_output}
            
            Focus on:
            - Completeness and coverage
            - Process optimization
            - Systematic improvements
            - Structured enhancements
            
            Provide systematic reflection in JSON format.
            """,
            
            "intuitive": f"""
            Provide an intuitive and experiential perspective on the output:
            Task: {state.task}
            Output: {state.current_output}
            
            Focus on:
            - User experience and intuition
            - Emotional resonance
            - Practical usability
            - Intuitive improvements
            
            Provide intuitive reflection in JSON format.
            """,
            
            "critical": f"""
            Critically evaluate the current output with rigorous scrutiny:
            Task: {state.task}
            Output: {state.current_output}
            
            Focus on:
            - Potential flaws and weaknesses
            - Critical assumptions
            - Risk assessment
            - Robust improvements
            
            Provide critical reflection in JSON format.
            """
        }
        
        strategy_reflections = {}
        
        # Generate reflections concurrently
        tasks = []
        for strategy, prompt in strategy_prompts.items():
            tasks.append(self._generate_strategy_reflection(strategy, prompt, llm_provider))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for strategy, result in zip(self.strategies, results):
            if isinstance(result, Exception):
                self.logger.warning(f"Strategy {strategy} failed: {result}")
                strategy_reflections[strategy] = {"error": str(result), "improvements": []}
            else:
                strategy_reflections[strategy] = result
        
        return strategy_reflections
    
    async def _generate_strategy_reflection(
        self, strategy: str, prompt: str, llm_provider
    ) -> Dict[str, Any]:
        """Generate reflection for a specific strategy."""
        
        try:
            response = await llm_provider.generate_async(prompt)
            reflection_data = json.loads(response)
            reflection_data["strategy"] = strategy
            reflection_data["confidence"] = random.uniform(0.6, 0.9)  # Simulated confidence
            return reflection_data
        except json.JSONDecodeError:
            return {
                "strategy": strategy,
                "improvements": [f"{strategy}_improvement"],
                "confidence": 0.5,
                "reasoning": f"{strategy}_based_analysis"
            }
    
    def _weight_strategy_reflections(
        self, strategy_reflections: Dict[str, Dict], state: ReflexionState
    ) -> Dict[str, Dict[str, Any]]:
        """Apply weights to strategy reflections based on historical performance."""
        
        weighted_reflections = {}
        
        for strategy, reflection in strategy_reflections.items():
            weight = self.strategy_weights.get(strategy, 1.0)
            confidence = reflection.get("confidence", 0.5)
            
            # Combine weight and confidence
            effective_weight = weight * confidence
            
            weighted_reflections[strategy] = {
                **reflection,
                "effective_weight": effective_weight,
                "base_weight": weight,
                "confidence": confidence
            }
        
        return weighted_reflections
    
    def _ensemble_voting(self, weighted_reflections: Dict[str, Dict]) -> Dict[str, Any]:
        """Use ensemble voting to determine consensus improvements."""
        
        # Collect all improvements with weights
        improvement_votes = defaultdict(float)
        total_weight = 0
        
        for strategy, reflection in weighted_reflections.items():
            weight = reflection["effective_weight"]
            total_weight += weight
            
            improvements = reflection.get("improvements", [])
            for improvement in improvements:
                improvement_votes[improvement] += weight
        
        # Normalize votes
        normalized_votes = {
            improvement: votes / total_weight
            for improvement, votes in improvement_votes.items()
        }
        
        # Select improvements above threshold
        consensus_improvements = [
            improvement for improvement, vote_share in normalized_votes.items()
            if vote_share >= self.voting_threshold
        ]
        
        # If no consensus, take top weighted improvements
        if not consensus_improvements:
            sorted_improvements = sorted(
                normalized_votes.items(), key=lambda x: x[1], reverse=True
            )
            consensus_improvements = [imp for imp, _ in sorted_improvements[:3]]
        
        return {
            "consensus_improvements": consensus_improvements,
            "vote_distribution": normalized_votes,
            "total_strategies": len(weighted_reflections),
            "voting_threshold": self.voting_threshold,
            "strategy_contributions": {
                strategy: reflection["effective_weight"] / total_weight
                for strategy, reflection in weighted_reflections.items()
            }
        }
    
    async def _apply_ensemble_improvements(
        self, state: ReflexionState, consensus: Dict[str, Any], llm_provider
    ) -> ReflexionState:
        """Apply ensemble-determined improvements."""
        
        improvement_prompt = f"""
        Apply the following consensus improvements from ensemble analysis:
        
        Current Output: {state.current_output}
        Consensus Improvements: {consensus['consensus_improvements']}
        Strategy Contributions: {consensus['strategy_contributions']}
        
        Integrate these improvements while maintaining coherence and quality.
        Focus on the highest-voted improvements: {consensus['consensus_improvements'][:3]}
        
        Provide an enhanced version that incorporates the ensemble wisdom.
        """
        
        improved_output = await llm_provider.generate_async(improvement_prompt)
        
        # Update state with ensemble information
        new_state = ReflexionState(
            iteration=state.iteration + 1,
            task=state.task,
            current_output=improved_output,
            historical_outputs=state.historical_outputs + [state.current_output],
            success_scores=state.success_scores,
            reflections=state.reflections,
            meta_reflections=state.meta_reflections + [consensus],
            strategy_history=state.strategy_history + ["ensemble"]
        )
        
        return new_state
    
    def _adjust_strategy_weights(
        self, strategy_reflections: Dict[str, Dict], improved_state: ReflexionState
    ):
        """Adaptively adjust strategy weights based on performance."""
        
        # Simplified weight adjustment based on improvement quality
        if len(improved_state.historical_outputs) > 0:
            # Measure improvement (simplified metric)
            improvement_score = len(improved_state.current_output) / max(
                len(improved_state.historical_outputs[-1]), 1
            )
            
            # Adjust weights based on contribution and performance
            for strategy, reflection in strategy_reflections.items():
                current_weight = self.strategy_weights[strategy]
                contribution = reflection.get("effective_weight", 0.5)
                
                # Reward strategies that contributed to improvement
                if improvement_score > 1.0:  # Improvement detected
                    adjustment = 0.1 * contribution
                    self.strategy_weights[strategy] = min(2.0, current_weight + adjustment)
                elif improvement_score < 0.9:  # Degradation detected
                    adjustment = 0.05 * contribution
                    self.strategy_weights[strategy] = max(0.5, current_weight - adjustment)
    
    def _evaluate_ensemble_success(
        self, state: ReflexionState, consensus: Dict[str, Any]
    ) -> bool:
        """Evaluate success based on ensemble criteria."""
        
        success_criteria = 0
        total_criteria = 3
        
        # Consensus strength
        if len(consensus["consensus_improvements"]) >= 2:
            success_criteria += 1
        
        # Output improvement
        if len(state.historical_outputs) > 0:
            if len(state.current_output) > len(state.historical_outputs[-1]):
                success_criteria += 1
        
        # Strategy diversity
        if len(consensus["strategy_contributions"]) >= 3:
            success_criteria += 1
        
        return success_criteria >= 2


class QuantumInspiredReflexionAlgorithm:
    """Quantum-inspired reflexion using superposition and entanglement concepts."""
    
    def __init__(self, superposition_states: int = 5, entanglement_strength: float = 0.7):
        self.superposition_states = superposition_states
        self.entanglement_strength = entanglement_strength
        self.quantum_state = None
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, state: ReflexionState, llm_provider) -> Tuple[bool, ReflexionState]:
        """Execute quantum-inspired reflexion algorithm."""
        
        # Initialize quantum superposition of reflection states
        superposition_states = await self._create_superposition_states(state, llm_provider)
        
        # Apply quantum entanglement between improvement strategies
        entangled_strategies = self._create_entangled_strategies(superposition_states)
        
        # Quantum interference for optimization
        optimized_reflections = self._apply_quantum_interference(entangled_strategies)
        
        # Collapse superposition through measurement (evaluation)
        collapsed_state = await self._collapse_superposition(
            optimized_reflections, state, llm_provider
        )
        
        # Evaluate quantum success
        success = self._evaluate_quantum_success(collapsed_state, superposition_states)
        
        return success, collapsed_state
    
    async def _create_superposition_states(
        self, state: ReflexionState, llm_provider
    ) -> List[Dict[str, Any]]:
        """Create superposition of possible reflection states."""
        
        superposition_prompts = []
        
        for i in range(self.superposition_states):
            # Vary the reflection focus using quantum-inspired randomness
            focus_areas = ["accuracy", "creativity", "efficiency", "completeness", "innovation"]
            primary_focus = random.choice(focus_areas)
            secondary_focus = random.choice([f for f in focus_areas if f != primary_focus])
            
            quantum_angle = (2 * math.pi * i) / self.superposition_states  # Quantum phase
            amplitude = math.cos(quantum_angle)  # Quantum amplitude
            
            prompt = f"""
            Analyze and reflect on the current output with quantum state {i}:
            
            Task: {state.task}
            Output: {state.current_output}
            Primary Focus: {primary_focus} (amplitude: {amplitude:.3f})
            Secondary Focus: {secondary_focus}
            Quantum Phase: {quantum_angle:.3f}
            
            Generate reflection considering quantum superposition of possibilities.
            Focus on probabilistic improvements and uncertainty quantification.
            
            Provide analysis in JSON format including confidence intervals.
            """
            
            superposition_prompts.append((i, prompt, amplitude, quantum_angle))
        
        # Generate superposition states concurrently
        tasks = [
            self._generate_quantum_state(i, prompt, amplitude, phase, llm_provider)
            for i, prompt, amplitude, phase in superposition_prompts
        ]
        
        superposition_states = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful states
        valid_states = [
            s for s in superposition_states
            if not isinstance(s, Exception)
        ]
        
        return valid_states
    
    async def _generate_quantum_state(
        self, state_id: int, prompt: str, amplitude: float, phase: float, llm_provider
    ) -> Dict[str, Any]:
        """Generate a single quantum reflection state."""
        
        try:
            response = await llm_provider.generate_async(prompt)
            state_data = json.loads(response)
            
            return {
                "state_id": state_id,
                "amplitude": amplitude,
                "phase": phase,
                "reflection": state_data,
                "confidence": state_data.get("confidence", random.uniform(0.5, 0.9)),
                "improvements": state_data.get("improvements", [f"quantum_improvement_{state_id}"]),
                "uncertainty": random.uniform(0.1, 0.3)  # Quantum uncertainty
            }
        except (json.JSONDecodeError, Exception) as e:
            return {
                "state_id": state_id,
                "amplitude": amplitude,
                "phase": phase,
                "reflection": {"error": str(e)},
                "confidence": 0.3,
                "improvements": [f"quantum_fallback_{state_id}"],
                "uncertainty": 0.5
            }
    
    def _create_entangled_strategies(self, superposition_states: List[Dict]) -> List[Dict[str, Any]]:
        """Create quantum entanglement between improvement strategies."""
        
        entangled_strategies = []
        
        for i, state1 in enumerate(superposition_states):
            for j, state2 in enumerate(superposition_states[i+1:], i+1):
                # Calculate entanglement correlation
                phase_diff = abs(state1["phase"] - state2["phase"])
                amplitude_product = state1["amplitude"] * state2["amplitude"]
                
                # Entanglement strength based on phase correlation
                entanglement = self.entanglement_strength * math.cos(phase_diff) * amplitude_product
                
                if abs(entanglement) > 0.3:  # Significant entanglement threshold
                    entangled_strategies.append({
                        "state_pair": (state1["state_id"], state2["state_id"]),
                        "entanglement_strength": entanglement,
                        "combined_improvements": (
                            state1["improvements"] + state2["improvements"]
                        ),
                        "combined_confidence": (
                            state1["confidence"] * state2["confidence"]
                        ) ** 0.5,
                        "interference_pattern": self._calculate_interference(
                            state1, state2, entanglement
                        )
                    })
        
        return entangled_strategies
    
    def _calculate_interference(
        self, state1: Dict, state2: Dict, entanglement: float
    ) -> Dict[str, Any]:
        """Calculate quantum interference pattern between states."""
        
        # Constructive vs destructive interference
        phase_diff = abs(state1["phase"] - state2["phase"])
        
        if phase_diff < math.pi / 2:
            interference_type = "constructive"
            enhancement_factor = 1 + abs(entanglement)
        else:
            interference_type = "destructive"  
            enhancement_factor = 1 - abs(entanglement) * 0.5
        
        return {
            "type": interference_type,
            "enhancement_factor": enhancement_factor,
            "phase_difference": phase_diff,
            "probability_amplitude": abs(entanglement) * enhancement_factor
        }
    
    def _apply_quantum_interference(self, entangled_strategies: List[Dict]) -> List[Dict[str, Any]]:
        """Apply quantum interference to optimize reflection strategies."""
        
        optimized_reflections = []
        
        for strategy in entangled_strategies:
            interference = strategy["interference_pattern"]
            
            # Apply interference effects
            if interference["type"] == "constructive":
                # Amplify improvements
                enhanced_improvements = []
                for imp in strategy["combined_improvements"]:
                    enhanced_improvements.append(f"enhanced_{imp}")
                
                optimized_reflections.append({
                    "improvements": enhanced_improvements,
                    "confidence": min(1.0, strategy["combined_confidence"] * interference["enhancement_factor"]),
                    "quantum_effect": "constructive_interference",
                    "probability": interference["probability_amplitude"],
                    "entanglement_strength": strategy["entanglement_strength"]
                })
            
            elif interference["type"] == "destructive" and interference["enhancement_factor"] > 0.7:
                # Keep partially destructive patterns that still add value
                filtered_improvements = strategy["combined_improvements"][::2]  # Take every other
                
                optimized_reflections.append({
                    "improvements": filtered_improvements,
                    "confidence": strategy["combined_confidence"] * interference["enhancement_factor"],
                    "quantum_effect": "filtered_destructive_interference",
                    "probability": interference["probability_amplitude"],
                    "entanglement_strength": strategy["entanglement_strength"]
                })
        
        # Sort by probability amplitude (quantum measurement likelihood)
        optimized_reflections.sort(key=lambda x: x["probability"], reverse=True)
        
        return optimized_reflections[:3]  # Top 3 quantum states
    
    async def _collapse_superposition(
        self, optimized_reflections: List[Dict], state: ReflexionState, llm_provider
    ) -> ReflexionState:
        """Collapse quantum superposition through measurement (final reflection)."""
        
        # Quantum measurement - select highest probability state
        if optimized_reflections:
            dominant_reflection = optimized_reflections[0]
        else:
            # Fallback quantum state
            dominant_reflection = {
                "improvements": ["quantum_fallback"],
                "confidence": 0.5,
                "quantum_effect": "collapsed_superposition"
            }
        
        collapse_prompt = f"""
        Apply quantum-collapsed improvements to enhance the output:
        
        Current Output: {state.current_output}
        Quantum Improvements: {dominant_reflection['improvements']}
        Quantum Confidence: {dominant_reflection['confidence']}
        Quantum Effect: {dominant_reflection['quantum_effect']}
        
        Integrate these quantum-optimized enhancements while maintaining coherence.
        Consider the probabilistic nature and uncertainty in the improvements.
        
        Provide the quantum-enhanced output.
        """
        
        collapsed_output = await llm_provider.generate_async(collapse_prompt)
        
        # Update state with quantum information
        new_state = ReflexionState(
            iteration=state.iteration + 1,
            task=state.task,
            current_output=collapsed_output,
            historical_outputs=state.historical_outputs + [state.current_output],
            success_scores=state.success_scores,
            reflections=state.reflections,
            meta_reflections=state.meta_reflections + [{
                "quantum_algorithm": "quantum_inspired_reflexion",
                "superposition_states": len(optimized_reflections),
                "dominant_reflection": dominant_reflection,
                "quantum_collapse": True
            }],
            strategy_history=state.strategy_history + ["quantum"]
        )
        
        return new_state
    
    def _evaluate_quantum_success(
        self, collapsed_state: ReflexionState, superposition_states: List[Dict]
    ) -> bool:
        """Evaluate success using quantum-inspired criteria."""
        
        success_probability = 0.0
        
        # Quantum coherence: multiple states contributed
        if len(superposition_states) >= 3:
            success_probability += 0.3
        
        # Quantum enhancement: output improved
        if len(collapsed_state.historical_outputs) > 0:
            improvement_ratio = len(collapsed_state.current_output) / max(
                len(collapsed_state.historical_outputs[-1]), 1
            )
            if improvement_ratio > 1.0:
                success_probability += 0.4
        
        # Quantum confidence: high-confidence states present
        avg_confidence = np.mean([s.get("confidence", 0.5) for s in superposition_states])
        if avg_confidence > 0.7:
            success_probability += 0.3
        
        # Quantum measurement: probabilistic success
        quantum_threshold = random.uniform(0.6, 0.8)
        return success_probability >= quantum_threshold


class ResearchComparator:
    """Comprehensive comparison framework for reflexion algorithms."""
    
    def __init__(self):
        self.algorithms = {
            ReflexionAlgorithm.HIERARCHICAL_REFLEXION: HierarchicalReflexionAlgorithm(),
            ReflexionAlgorithm.ENSEMBLE_REFLEXION: EnsembleReflexionAlgorithm(),
            ReflexionAlgorithm.QUANTUM_INSPIRED_REFLEXION: QuantumInspiredReflexionAlgorithm()
        }
        self.performance_data: Dict[ReflexionAlgorithm, List[AlgorithmPerformance]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    async def comparative_study(
        self,
        test_tasks: List[str],
        llm_provider,
        iterations_per_algorithm: int = 10,
        max_iterations_per_task: int = 3
    ) -> Dict[str, Any]:
        """Conduct comprehensive comparative study of reflexion algorithms."""
        
        study_results = {
            "study_metadata": {
                "test_tasks": len(test_tasks),
                "algorithms": len(self.algorithms),
                "iterations_per_algorithm": iterations_per_algorithm,
                "max_iterations_per_task": max_iterations_per_task,
                "study_timestamp": datetime.now().isoformat()
            },
            "algorithm_performances": {},
            "statistical_analysis": {},
            "comparative_insights": {}
        }
        
        # Run comparative tests
        for algorithm_type, algorithm in self.algorithms.items():
            self.logger.info(f"Testing algorithm: {algorithm_type.value}")
            
            algorithm_results = []
            
            for task in test_tasks:
                for iteration in range(iterations_per_algorithm):
                    # Run single algorithm test
                    result = await self._run_algorithm_test(
                        algorithm, algorithm_type, task, llm_provider, max_iterations_per_task
                    )
                    algorithm_results.append(result)
            
            # Calculate performance metrics
            performance = self._calculate_algorithm_performance(algorithm_type, algorithm_results)
            study_results["algorithm_performances"][algorithm_type.value] = performance
        
        # Statistical analysis
        study_results["statistical_analysis"] = self._perform_statistical_analysis(
            study_results["algorithm_performances"]
        )
        
        # Comparative insights
        study_results["comparative_insights"] = self._generate_comparative_insights(
            study_results["algorithm_performances"],
            study_results["statistical_analysis"]
        )
        
        return study_results
    
    async def _run_algorithm_test(
        self,
        algorithm,
        algorithm_type: ReflexionAlgorithm,
        task: str,
        llm_provider,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Run a single algorithm test."""
        
        start_time = time.time()
        
        # Initialize state
        initial_output = f"Initial approach to: {task}"
        state = ReflexionState(
            iteration=0,
            task=task,
            current_output=initial_output,
            historical_outputs=[],
            success_scores=[],
            reflections=[],
            meta_reflections=[]
        )
        
        # Run algorithm iterations
        iteration_results = []
        success_achieved = False
        
        for i in range(max_iterations):
            iteration_start = time.time()
            
            try:
                success, new_state = await algorithm.execute(state, llm_provider)
                iteration_time = time.time() - iteration_start
                
                iteration_results.append({
                    "iteration": i + 1,
                    "success": success,
                    "execution_time": iteration_time,
                    "output_length": len(new_state.current_output),
                    "reflections_generated": len(new_state.meta_reflections) - len(state.meta_reflections)
                })
                
                state = new_state
                
                if success:
                    success_achieved = True
                    break
                    
            except Exception as e:
                self.logger.error(f"Algorithm {algorithm_type.value} failed on iteration {i}: {e}")
                iteration_results.append({
                    "iteration": i + 1,
                    "success": False,
                    "execution_time": time.time() - iteration_start,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        
        return {
            "algorithm": algorithm_type.value,
            "task": task,
            "success_achieved": success_achieved,
            "total_iterations": len(iteration_results),
            "total_time": total_time,
            "iteration_results": iteration_results,
            "final_output_length": len(state.current_output),
            "total_reflections": len(state.meta_reflections)
        }
    
    def _calculate_algorithm_performance(
        self, algorithm_type: ReflexionAlgorithm, results: List[Dict]
    ) -> AlgorithmPerformance:
        """Calculate comprehensive performance metrics for an algorithm."""
        
        # Basic metrics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success_achieved"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Iteration metrics
        avg_iterations = np.mean([r["total_iterations"] for r in results])
        
        # Time metrics  
        avg_execution_time = np.mean([r["total_time"] for r in results])
        
        # Quality metrics (simplified)
        avg_reflection_quality = np.mean([
            r["total_reflections"] / max(r["total_iterations"], 1) for r in results
        ])
        
        # Convergence rate (how quickly success is achieved)
        successful_results = [r for r in results if r["success_achieved"]]
        if successful_results:
            convergence_rate = 1.0 / np.mean([r["total_iterations"] for r in successful_results])
        else:
            convergence_rate = 0.0
        
        # Stability (consistency of performance)
        time_std = np.std([r["total_time"] for r in results])
        stability_score = 1.0 / (1.0 + time_std)  # Higher stability = lower variance
        
        # Adaptability (performance across different tasks)
        task_success_rates = {}
        for result in results:
            task = result["task"]
            if task not in task_success_rates:
                task_success_rates[task] = []
            task_success_rates[task].append(result["success_achieved"])
        
        task_variance = np.var([
            np.mean(successes) for successes in task_success_rates.values()
        ]) if task_success_rates else 0
        adaptability_score = 1.0 / (1.0 + task_variance)
        
        # Confidence interval (simplified)
        confidence_interval = (
            max(0, success_rate - 1.96 * np.sqrt(success_rate * (1-success_rate) / total_tests)),
            min(1, success_rate + 1.96 * np.sqrt(success_rate * (1-success_rate) / total_tests))
        )
        
        # Statistical significance (simplified z-score)
        statistical_significance = abs(success_rate - 0.5) / np.sqrt(0.25 / total_tests)
        
        return AlgorithmPerformance(
            algorithm=algorithm_type,
            success_rate=success_rate,
            avg_iterations=avg_iterations,
            avg_execution_time=avg_execution_time,
            avg_reflection_quality=avg_reflection_quality,
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            adaptability_score=adaptability_score,
            sample_size=total_tests,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance
        )
    
    def _perform_statistical_analysis(self, performances: Dict[str, AlgorithmPerformance]) -> Dict[str, Any]:
        """Perform statistical analysis across algorithms."""
        
        # Extract metrics for comparison
        algorithms = list(performances.keys())
        success_rates = [performances[alg].success_rate for alg in algorithms]
        execution_times = [performances[alg].avg_execution_time for alg in algorithms]
        iterations = [performances[alg].avg_iterations for alg in algorithms]
        
        # Statistical tests (simplified implementations)
        analysis = {
            "sample_sizes": {alg: performances[alg].sample_size for alg in algorithms},
            
            "success_rate_analysis": {
                "mean": np.mean(success_rates),
                "std": np.std(success_rates),
                "range": (min(success_rates), max(success_rates)),
                "best_algorithm": algorithms[np.argmax(success_rates)],
                "worst_algorithm": algorithms[np.argmin(success_rates)]
            },
            
            "execution_time_analysis": {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "range": (min(execution_times), max(execution_times)),
                "fastest_algorithm": algorithms[np.argmin(execution_times)],
                "slowest_algorithm": algorithms[np.argmax(execution_times)]
            },
            
            "iteration_efficiency": {
                "mean": np.mean(iterations),
                "std": np.std(iterations),
                "most_efficient": algorithms[np.argmin(iterations)],
                "least_efficient": algorithms[np.argmax(iterations)]
            },
            
            "overall_rankings": self._calculate_overall_rankings(performances),
            
            "statistical_significance": {
                alg: performances[alg].statistical_significance > 1.96
                for alg in algorithms  # z > 1.96 indicates significance at p < 0.05
            }
        }
        
        return analysis
    
    def _calculate_overall_rankings(self, performances: Dict[str, AlgorithmPerformance]) -> Dict[str, Any]:
        """Calculate overall algorithm rankings using multiple criteria."""
        
        algorithms = list(performances.keys())
        
        # Normalize metrics to 0-1 scale for comparison
        metrics = {}
        for metric_name in ["success_rate", "convergence_rate", "stability_score", "adaptability_score"]:
            values = [getattr(performances[alg], metric_name) for alg in algorithms]
            max_val, min_val = max(values), min(values)
            
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized = [1.0] * len(values)  # All equal
            
            metrics[metric_name] = dict(zip(algorithms, normalized))
        
        # Reverse normalize execution time and iterations (lower is better)
        for metric_name in ["avg_execution_time", "avg_iterations"]:
            values = [getattr(performances[alg], metric_name) for alg in algorithms]
            max_val, min_val = max(values), min(values)
            
            if max_val > min_val:
                normalized = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                normalized = [1.0] * len(values)
            
            metrics[metric_name] = dict(zip(algorithms, normalized))
        
        # Calculate weighted composite scores
        weights = {
            "success_rate": 0.3,
            "avg_execution_time": 0.2,
            "avg_iterations": 0.15,
            "convergence_rate": 0.15,
            "stability_score": 0.1,
            "adaptability_score": 0.1
        }
        
        composite_scores = {}
        for alg in algorithms:
            score = sum(
                weights[metric] * metrics[metric][alg]
                for metric in weights.keys()
            )
            composite_scores[alg] = score
        
        # Rank algorithms
        ranked_algorithms = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "composite_scores": composite_scores,
            "rankings": {alg: rank + 1 for rank, (alg, _) in enumerate(ranked_algorithms)},
            "top_algorithm": ranked_algorithms[0][0],
            "performance_gaps": {
                alg: ranked_algorithms[0][1] - score
                for alg, score in ranked_algorithms[1:]
            }
        }
    
    def _generate_comparative_insights(
        self, performances: Dict[str, AlgorithmPerformance], statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights from comparative analysis."""
        
        insights = {
            "key_findings": [],
            "algorithm_strengths": {},
            "algorithm_weaknesses": {},
            "recommendations": {},
            "future_research_directions": []
        }
        
        # Key findings
        best_overall = statistics["overall_rankings"]["top_algorithm"]
        best_success_rate = statistics["success_rate_analysis"]["best_algorithm"]
        fastest = statistics["execution_time_analysis"]["fastest_algorithm"]
        
        insights["key_findings"].extend([
            f"Best overall performance: {best_overall}",
            f"Highest success rate: {best_success_rate} ({performances[best_success_rate].success_rate:.1%})",
            f"Fastest execution: {fastest} ({performances[fastest].avg_execution_time:.2f}s avg)",
            f"Performance gap between top algorithms: {max(statistics['overall_rankings']['performance_gaps'].values()):.3f}"
        ])
        
        # Algorithm-specific insights
        for alg_name, perf in performances.items():
            strengths = []
            weaknesses = []
            
            # Identify strengths (top 33% performance)
            if perf.success_rate >= np.percentile([p.success_rate for p in performances.values()], 67):
                strengths.append("High success rate")
            
            if perf.convergence_rate >= np.percentile([p.convergence_rate for p in performances.values()], 67):
                strengths.append("Fast convergence")
            
            if perf.stability_score >= np.percentile([p.stability_score for p in performances.values()], 67):
                strengths.append("Consistent performance")
            
            if perf.adaptability_score >= np.percentile([p.adaptability_score for p in performances.values()], 67):
                strengths.append("Good task adaptability")
            
            # Identify weaknesses (bottom 33% performance)  
            if perf.avg_execution_time >= np.percentile([p.avg_execution_time for p in performances.values()], 67):
                weaknesses.append("Slower execution")
            
            if perf.avg_iterations >= np.percentile([p.avg_iterations for p in performances.values()], 67):
                weaknesses.append("Requires more iterations")
            
            insights["algorithm_strengths"][alg_name] = strengths
            insights["algorithm_weaknesses"][alg_name] = weaknesses
        
        # Recommendations
        insights["recommendations"] = {
            "for_accuracy": statistics["success_rate_analysis"]["best_algorithm"],
            "for_speed": statistics["execution_time_analysis"]["fastest_algorithm"],
            "for_efficiency": statistics["iteration_efficiency"]["most_efficient"],
            "for_robustness": max(
                performances.items(),
                key=lambda x: x[1].stability_score * x[1].adaptability_score
            )[0]
        }
        
        # Future research directions
        insights["future_research_directions"] = [
            "Hybrid algorithms combining strengths of top performers",
            "Adaptive algorithm selection based on task characteristics",
            "Multi-objective optimization balancing speed and accuracy",
            "Investigation of quantum-classical hybrid approaches",
            "Deep learning integration for reflection quality prediction"
        ]
        
        return insights


# Global research comparator instance
research_comparator = ResearchComparator()