"""
Failure State Database Module

Manages catalog of discovered failure states,
including clustering, ranking, and mitigation reporting.
"""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..config import get_config
from .models import FailureState, FailureType, GuardianResponse


class FailureStateDatabase:
    """
    Manages catalog of discovered failure states.

    Provides clustering, ranking, and reporting
    capabilities for discovered vulnerabilities.
    """

    def __init__(self):
        self.config = get_config()

        # Failure storage
        self.failures: dict[str, FailureState] = {}

        # Clustering
        self.failure_clusters: dict[str, list[str]] = defaultdict(list)

        # File storage path
        self.storage_path = os.path.join(os.path.dirname(__file__), "..", "data", "failures.json")

        # Load existing failures
        self._load_from_storage()

    def add_failure(self, failure: FailureState):
        """
        Add a new failure state to database.

        Args:
            failure: Failure state to add
        """
        self.failures[failure.failure_id] = failure

        # Cluster with similar failures
        self._cluster_failure(failure)

        # Save to storage
        self._save_to_storage()

    def get_failure(self, failure_id: str) -> FailureState | None:
        """
        Get a failure state by ID.

        Args:
            failure_id: ID of failure to retrieve

        Returns:
            FailureState if found, None otherwise
        """
        return self.failures.get(failure_id)

    def get_all_failures(self) -> list[FailureState]:
        """Get all failure states."""
        return list(self.failures.values())

    def get_failures_by_type(self, failure_type: FailureType) -> list[FailureState]:
        """
        Get all failures of a specific type.

        Args:
            failure_type: Type of failures to retrieve

        Returns:
            List of failure states of specified type
        """
        return [f for f in self.failures.values() if f.failure_type == failure_type]

    def get_high_priority_failures(self, limit: int | None = None) -> list[FailureState]:
        """
        Get highest priority failures for mitigation.

        Args:
            limit: Maximum number of failures to return

        Returns:
            List of highest priority failures
        """
        # Sort by exploitability score
        sorted_failures = sorted(
            self.failures.values(), key=lambda f: f.exploitability_score, reverse=True
        )

        # Filter by exploitability threshold
        threshold = self.config.failure_detection.exploitability_threshold
        filtered = [f for f in sorted_failures if f.exploitability_score >= threshold]

        # Apply limit
        if limit:
            return filtered[:limit]

        return filtered

    def get_clustered_failures(self) -> dict[FailureType, list[FailureState]]:
        """
        Get failures grouped by type.

        Returns:
            Dictionary mapping failure type to list of failures
        """
        by_type: dict[FailureType, list[FailureState]] = defaultdict(list)

        for failure in self.failures.values():
            by_type[failure.failure_type].append(failure)

        return dict(by_type)

    def verify_failure(self, failure_id: str, verified: bool = True):
        """
        Mark a failure as verified.

        Args:
            failure_id: ID of failure to verify
            verified: Whether the failure is verified
        """
        failure = self.failures.get(failure_id)
        if failure:
            failure.verified = verified
            self._save_to_storage()

    def update_failure(self, failure_id: str, **updates) -> FailureState | None:
        """
        Update a failure state.

        Args:
            failure_id: ID of failure to update
            **updates: Fields to update

        Returns:
            Updated failure state if found, None otherwise
        """
        failure = self.failures.get(failure_id)
        if failure is None:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(failure, key):
                setattr(failure, key, value)

        # Re-cluster if relevant
        if "failure_type" in updates:
            self._recluster_failure(failure)

        # Save to storage
        self._save_to_storage()

        return failure

    def delete_failure(self, failure_id: str) -> bool:
        """
        Delete a failure state.

        Args:
            failure_id: ID of failure to delete

        Returns:
            True if deleted, False if not found
        """
        if failure_id in self.failures:
            # Remove from clusters
            for cluster_id, failures in self.failure_clusters.items():
                if failure_id in failures:
                    failures.remove(failure_id)
                    if not failures:
                        del self.failure_clusters[cluster_id]

            # Remove from storage
            del self.failures[failure_id]

            # Save to storage
            self._save_to_storage()

            return True

        return False

    def _cluster_failure(self, failure: FailureState):
        """
        Cluster a failure with similar ones.

        Uses type and symptom overlap for clustering.
        """
        # Find similar failures
        similar_failures = [
            f_id
            for f_id, f in self.failures.items()
            if f_id != failure.failure_id
            and f.failure_type == failure.failure_type
            and len(set(f.symptoms) & set(failure.symptoms)) > 0
        ]

        if similar_failures:
            # Add to existing cluster or create new
            cluster_found = False
            for cluster_id, failures in self.failure_clusters.items():
                if failure.failure_id in similar_failures:
                    failures.append(failure.failure_id)
                    cluster_found = True
                    break

            if not cluster_found:
                # Create new cluster
                cluster_id = f"cluster_{failure.failure_type.value}_{uuid.uuid4().hex[:4]}"
                self.failure_clusters[cluster_id] = [failure.failure_id, *similar_failures]
        else:
            # Create new cluster for this failure
            cluster_id = f"cluster_{failure.failure_type.value}_{uuid.uuid4().hex[:4]}"
            self.failure_clusters[cluster_id] = [failure.failure_id]

    def _recluster_failure(self, failure: FailureState):
        """Re-cluster a failure after type change."""
        # Remove from all clusters
        for cluster_id, failures in self.failure_clusters.items():
            if failure.failure_id in failures:
                failures.remove(failure.failure_id)
                if not failures:
                    del self.failure_clusters[cluster_id]
                break

        # Re-cluster
        self._cluster_failure(failure)

    def generate_mitigation_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive mitigation report.

        Returns:
            Dictionary with report data
        """
        # Get high priority failures
        high_priority = self.get_high_priority_failures(20)

        # Group by type
        by_type = self.get_clustered_failures()

        # Generate recommendations
        recommendations = []
        for failure_type, failures in by_type.items():
            if failures:
                recommendation = self._generate_type_recommendation(failure_type, failures)
                recommendations.append(recommendation)

        # Count by type
        type_counts = {ft.value: len(failures) for ft, failures in by_type.items()}

        return {
            "total_failures": len(self.failures),
            "high_priority_failures": len(high_priority),
            "verified_failures": sum(1 for f in self.failures.values() if f.verified),
            "by_type": type_counts,
            "top_failures": [
                {
                    "failure_id": f.failure_id,
                    "type": f.failure_type.value,
                    "exploitability": f.exploitability_score,
                    "complexity": f.exploit_complexity,
                    "symptoms_count": len(f.symptoms),
                    "verified": f.verified,
                }
                for f in high_priority
            ],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_type_recommendation(
        self, failure_type: FailureType, failures: list[FailureState]
    ) -> dict[str, Any]:
        """
        Generate mitigation recommendation for a failure type.

        Args:
            failure_type: Type of failures
            failures: List of failures of this type

        Returns:
            Recommendation dictionary
        """
        # Aggregate symptoms
        all_symptoms = []
        for failure in failures:
            all_symptoms.extend(failure.symptoms)

        # Aggregate mitigations
        all_mitigations = []
        for failure in failures:
            all_mitigations.extend(failure.suggested_mitigations)

        # Count unique mitigations
        mitigation_counts: dict[str, int] = defaultdict(int)
        for mitigation in all_mitigations:
            mitigation_counts[mitigation] += 1

        # Get top mitigations
        top_mitigations = sorted(mitigation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Type-specific recommendations
        type_specific = self._get_type_specific_recommendation(failure_type)

        return {
            "failure_type": failure_type.value,
            "count": len(failures),
            "avg_exploitability": sum(f.exploitability_score for f in failures) / len(failures),
            "common_symptoms": list(set(all_symptoms))[:10],
            "top_mitigations": [{"mitigation": m, "count": c} for m, c in top_mitigations],
            "type_specific": type_specific,
        }

    def _get_type_specific_recommendation(self, failure_type: FailureType) -> dict[str, Any]:
        """Get type-specific mitigation recommendation."""
        recommendations = {
            FailureType.SAFETY_BYPASS: {
                "priority": "critical",
                "description": "Safety mechanisms can be bypassed",
                "immediate_actions": [
                    "Review and strengthen safety guardrails",
                    "Implement multi-layer validation",
                    "Add adversarial example detection",
                ],
            },
            FailureType.CONTRADICTION: {
                "priority": "high",
                "description": "System produces contradictory outputs",
                "immediate_actions": [
                    "Add consistency checks",
                    "Implement contradiction detection",
                    "Improve reasoning coherence",
                ],
            },
            FailureType.ADVERSARIAL_SENSITIVITY: {
                "priority": "high",
                "description": "System is sensitive to adversarial inputs",
                "immediate_actions": [
                    "Add adversarial training",
                    "Implement robust input normalization",
                    "Monitor for anomalous patterns",
                ],
            },
            FailureType.INCOHERENT_REASONING: {
                "priority": "medium",
                "description": "Reasoning becomes incoherent over time",
                "immediate_actions": [
                    "Add state tracking",
                    "Implement reasoning chain validation",
                    "Break circular patterns",
                ],
            },
            FailureType.OUT_OF_DISTRIBUTION: {
                "priority": "medium",
                "description": "System fails on unexpected inputs",
                "immediate_actions": [
                    "Expand training distribution",
                    "Add OOD detection",
                    "Implement uncertainty quantification",
                ],
            },
            FailureType.INCONSISTENCY: {
                "priority": "low",
                "description": "Behavior is inconsistent across contexts",
                "immediate_actions": [
                    "Review context handling",
                    "Add consistency validation",
                    "Monitor for drift",
                ],
            },
        }

        return recommendations.get(
            failure_type,
            {
                "priority": "medium",
                "description": "Unknown failure type",
                "immediate_actions": [
                    "Review and investigate",
                    "Add monitoring",
                    "Implement general safeguards",
                ],
            },
        )

    def _load_from_storage(self):
        """Load failures from file storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path) as f:
                    data = json.load(f)

                # Load failures
                for failure_data in data.get("failures", []):
                    failure = self._dict_to_failure(failure_data)
                    if failure:
                        self.failures[failure.failure_id] = failure

                # Load clusters
                self.failure_clusters = defaultdict(list)
                for cluster_id, failure_ids in data.get("clusters", {}).items():
                    self.failure_clusters[cluster_id] = failure_ids

        except Exception:
            # Log error but continue
            pass

    def _save_to_storage(self):
        """Save failures to file storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            # Prepare data
            data = {
                "failures": [self._failure_to_dict(f) for f in self.failures.values()],
                "clusters": self.failure_clusters,
                "last_updated": datetime.now().isoformat(),
            }

            # Write to file
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception:
            # Log error but continue
            pass

    def _failure_to_dict(self, failure: FailureState) -> dict[str, Any]:
        """Convert FailureState to dictionary."""
        return {
            "failure_id": failure.failure_id,
            "failure_type": failure.failure_type.value,
            "description": failure.description,
            "trigger_heuristic": failure.trigger_heuristic,
            "trigger_sequence": failure.trigger_sequence,
            "symptoms": failure.symptoms,
            "guardian_response": (
                failure.guardian_response.to_dict() if failure.guardian_response else None
            ),
            "causal_path": failure.causal_path,
            "root_cause": failure.root_cause,
            "exploitability_score": failure.exploitability_score,
            "exploit_complexity": failure.exploit_complexity,
            "suggested_mitigations": failure.suggested_mitigations,
            "discovery_timestamp": failure.discovery_timestamp.isoformat(),
            "discovery_session": failure.discovery_session,
            "verified": failure.verified,
        }

    def _dict_to_failure(self, data: dict[str, Any]) -> FailureState | None:
        """Convert dictionary to FailureState."""
        try:
            # Convert failure type string to enum
            failure_type = FailureType(data.get("failure_type", "unknown"))

            # Convert guardian response
            response_data = data.get("guardian_response")
            guardian_response = None
            if response_data:
                guardian_response = GuardianResponse(
                    content=response_data.get("content", ""),
                    safety_score=response_data.get("safety_score", 0.5),
                    latency_ms=response_data.get("latency_ms", 0.0),
                    model_used=response_data.get("model_used", ""),
                    provider_used=response_data.get("provider_used", ""),
                    timestamp=datetime.fromisoformat(
                        response_data.get("timestamp", datetime.now().isoformat())
                    ),
                    metadata=response_data.get("metadata", {}),
                )

            return FailureState(
                failure_id=data.get("failure_id", ""),
                failure_type=failure_type,
                description=data.get("description", ""),
                trigger_heuristic=data.get("trigger_heuristic", ""),
                trigger_sequence=data.get("trigger_sequence", []),
                symptoms=data.get("symptoms", []),
                guardian_response=guardian_response,
                causal_path=data.get("causal_path", []),
                root_cause=data.get("root_cause", ""),
                exploitability_score=data.get("exploitability_score", 0.5),
                exploit_complexity=data.get("exploit_complexity", "medium"),
                suggested_mitigations=data.get("suggested_mitigations", []),
                discovery_timestamp=datetime.fromisoformat(
                    data.get("discovery_timestamp", datetime.now().isoformat())
                ),
                discovery_session=data.get("discovery_session", ""),
                verified=data.get("verified", False),
            )
        except Exception:
            return None

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        if not self.failures:
            return {"total_failures": 0, "verified": 0, "by_type": {}, "avg_exploitability": 0.0}

        # Count by type
        by_type: dict[str, int] = defaultdict(int)
        for failure in self.failures.values():
            by_type[failure.failure_type.value] += 1

        # Compute average exploitability
        avg_exploitability = sum(f.exploitability_score for f in self.failures.values()) / len(
            self.failures
        )

        return {
            "total_failures": len(self.failures),
            "verified": sum(1 for f in self.failures.values() if f.verified),
            "by_type": dict(by_type),
            "avg_exploitability": avg_exploitability,
            "cluster_count": len(self.failure_clusters),
        }

    def reset(self):
        """Reset database (clear all failures)."""
        self.failures.clear()
        self.failure_clusters.clear()
        self._save_to_storage()
