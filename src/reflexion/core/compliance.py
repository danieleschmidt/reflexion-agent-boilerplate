"""Compliance and regulatory framework for reflexion agents."""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict

from .logging_config import logger
from .exceptions import ComplianceError


@dataclass
class ComplianceRecord:
    """Record for compliance tracking."""
    record_id: str
    timestamp: str
    action: str
    data_type: str
    user_id: Optional[str]
    data_categories: List[str]
    retention_period: int
    consent_given: bool
    purpose: str
    legal_basis: str
    hash_signature: str


@dataclass
class DataProcessingRecord:
    """GDPR Article 30 processing record."""
    processing_id: str
    controller_name: str
    purposes: List[str]
    categories_of_data_subjects: List[str]
    categories_of_personal_data: List[str]
    categories_of_recipients: List[str]
    retention_periods: Dict[str, int]
    security_measures: List[str]
    created_at: str
    updated_at: str


class GDPRCompliance:
    """GDPR compliance management system."""
    
    def __init__(self, data_dir: str = "./compliance_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.records_file = self.data_dir / "compliance_records.json"
        self.processing_records_file = self.data_dir / "processing_records.json"
        
        self.records: List[ComplianceRecord] = []
        self.processing_records: List[DataProcessingRecord] = []
        
        # Load existing records
        self._load_records()
        
        # GDPR data categories
        self.personal_data_categories = {
            "identity": ["name", "email", "user_id", "username"],
            "contact": ["email", "phone", "address"],
            "technical": ["ip_address", "session_id", "device_info"],
            "behavioral": ["usage_patterns", "preferences", "interactions"],
            "derived": ["ai_generated_content", "analysis_results"]
        }
        
        # Default retention periods (in days)
        self.default_retention = {
            "identity": 2555,  # 7 years
            "contact": 1095,   # 3 years
            "technical": 365,  # 1 year
            "behavioral": 730, # 2 years
            "derived": 365     # 1 year
        }
    
    def record_data_processing(
        self,
        action: str,
        data_content: str,
        user_id: Optional[str] = None,
        purpose: str = "reflexion_processing",
        legal_basis: str = "legitimate_interest",
        consent_given: bool = False
    ) -> str:
        """Record data processing activity for GDPR compliance."""
        
        # Analyze data content for personal data
        data_categories = self._analyze_data_categories(data_content)
        data_type = "personal_data" if data_categories else "non_personal_data"
        
        # Calculate retention period based on data categories
        retention_period = self._calculate_retention_period(data_categories)
        
        # Create compliance record
        record_id = self._generate_record_id()
        record = ComplianceRecord(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            action=action,
            data_type=data_type,
            user_id=user_id,
            data_categories=data_categories,
            retention_period=retention_period,
            consent_given=consent_given,
            purpose=purpose,
            legal_basis=legal_basis,
            hash_signature=self._create_data_hash(data_content)
        )
        
        self.records.append(record)
        self._save_records()
        
        logger.info(f"Recorded data processing: {action} for {data_type}")
        return record_id
    
    def request_data_deletion(self, user_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 17 - Right to erasure request."""
        user_records = [r for r in self.records if r.user_id == user_id]
        
        if not user_records:
            return {
                "status": "no_data_found",
                "message": f"No data found for user {user_id}",
                "records_deleted": 0
            }
        
        # Check if deletion is legally required
        deletable_records = []
        protected_records = []
        
        for record in user_records:
            if self._can_delete_record(record):
                deletable_records.append(record)
            else:
                protected_records.append(record)
        
        # Remove deletable records
        for record in deletable_records:
            self.records.remove(record)
        
        # Log deletion request
        deletion_record = ComplianceRecord(
            record_id=self._generate_record_id(),
            timestamp=datetime.now().isoformat(),
            action="data_deletion_request",
            data_type="deletion_log",
            user_id=user_id,
            data_categories=["deletion_request"],
            retention_period=2555,  # 7 years for legal records
            consent_given=True,
            purpose="gdpr_compliance",
            legal_basis="legal_obligation",
            hash_signature=self._create_data_hash(f"deletion_request_{user_id}")
        )
        
        self.records.append(deletion_record)
        self._save_records()
        
        return {
            "status": "completed",
            "records_deleted": len(deletable_records),
            "records_protected": len(protected_records),
            "protected_reasons": [self._get_protection_reason(r) for r in protected_records],
            "deletion_record_id": deletion_record.record_id
        }
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 15 - Right of access request."""
        user_records = [r for r in self.records if r.user_id == user_id]
        
        if not user_records:
            return {
                "status": "no_data_found",
                "message": f"No data found for user {user_id}",
                "export": {}
            }
        
        # Structure data export
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "data_categories": {},
            "processing_activities": [],
            "retention_information": {},
            "rights_information": {
                "right_to_access": "You can request access to your personal data",
                "right_to_rectification": "You can request correction of your personal data",
                "right_to_erasure": "You can request deletion of your personal data",
                "right_to_portability": "You can request transfer of your personal data",
                "right_to_object": "You can object to processing of your personal data"
            }
        }
        
        # Group records by data categories
        for record in user_records:
            for category in record.data_categories:
                if category not in export_data["data_categories"]:
                    export_data["data_categories"][category] = []
                
                export_data["data_categories"][category].append({
                    "record_id": record.record_id,
                    "timestamp": record.timestamp,
                    "action": record.action,
                    "purpose": record.purpose,
                    "legal_basis": record.legal_basis
                })
            
            # Add processing activity
            export_data["processing_activities"].append({
                "activity": record.action,
                "timestamp": record.timestamp,
                "purpose": record.purpose,
                "legal_basis": record.legal_basis,
                "retention_period_days": record.retention_period
            })
            
            # Add retention information
            for category in record.data_categories:
                export_data["retention_information"][category] = record.retention_period
        
        logger.info(f"Exported user data for: {user_id}")
        return {
            "status": "completed",
            "export": export_data
        }
    
    def audit_compliance_status(self) -> Dict[str, Any]:
        """Generate comprehensive compliance audit report."""
        now = datetime.now()
        
        # Analyze records
        total_records = len(self.records)
        personal_data_records = len([r for r in self.records if r.data_type == "personal_data"])
        records_with_consent = len([r for r in self.records if r.consent_given])
        
        # Check for expired data
        expired_records = []
        for record in self.records:
            record_date = datetime.fromisoformat(record.timestamp)
            retention_end = record_date + timedelta(days=record.retention_period)
            
            if now > retention_end:
                expired_records.append(record)
        
        # Legal basis analysis
        legal_basis_counts = {}
        for record in self.records:
            basis = record.legal_basis
            legal_basis_counts[basis] = legal_basis_counts.get(basis, 0) + 1
        
        # Generate audit report
        audit_report = {
            "audit_timestamp": now.isoformat(),
            "summary": {
                "total_records": total_records,
                "personal_data_records": personal_data_records,
                "consent_rate": (records_with_consent / max(total_records, 1)) * 100,
                "expired_records": len(expired_records),
                "compliance_score": self._calculate_compliance_score()
            },
            "legal_basis_distribution": legal_basis_counts,
            "expired_records": [
                {
                    "record_id": r.record_id,
                    "user_id": r.user_id,
                    "age_days": (now - datetime.fromisoformat(r.timestamp)).days,
                    "retention_period": r.retention_period
                }
                for r in expired_records[:10]  # First 10 expired records
            ],
            "recommendations": self._generate_compliance_recommendations(expired_records)
        }
        
        logger.info(f"Generated compliance audit report - Score: {audit_report['summary']['compliance_score']:.1f}%")
        return audit_report
    
    def _analyze_data_categories(self, data_content: str) -> List[str]:
        """Analyze data content to identify personal data categories."""
        categories = []
        data_lower = data_content.lower()
        
        for category, keywords in self.personal_data_categories.items():
            if any(keyword in data_lower for keyword in keywords):
                categories.append(category)
        
        # Additional heuristics
        if "@" in data_content:
            categories.append("contact")
        
        if any(char.isdigit() for char in data_content) and len(data_content) > 50:
            categories.append("technical")
        
        return list(set(categories))
    
    def _calculate_retention_period(self, categories: List[str]) -> int:
        """Calculate appropriate retention period based on data categories."""
        if not categories:
            return 365  # Default 1 year for non-personal data
        
        # Use the longest retention period among categories
        return max(self.default_retention.get(cat, 365) for cat in categories)
    
    def _can_delete_record(self, record: ComplianceRecord) -> bool:
        """Determine if a record can be legally deleted."""
        # Check if within retention period
        record_date = datetime.fromisoformat(record.timestamp)
        retention_end = record_date + timedelta(days=record.retention_period)
        
        if datetime.now() < retention_end:
            # Still within retention period - check for legal obligations
            protected_purposes = ["legal_obligation", "vital_interests", "public_task"]
            if record.legal_basis in protected_purposes:
                return False
        
        return True
    
    def _get_protection_reason(self, record: ComplianceRecord) -> str:
        """Get reason why a record is protected from deletion."""
        if record.legal_basis == "legal_obligation":
            return "Required for legal compliance"
        elif record.legal_basis == "vital_interests":
            return "Required for vital interests"
        elif record.legal_basis == "public_task":
            return "Required for public task"
        else:
            return "Within retention period"
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        if not self.records:
            return 100.0
        
        score = 100.0
        now = datetime.now()
        
        # Deduct points for expired data
        expired_count = 0
        for record in self.records:
            record_date = datetime.fromisoformat(record.timestamp)
            retention_end = record_date + timedelta(days=record.retention_period)
            
            if now > retention_end:
                expired_count += 1
        
        if expired_count > 0:
            score -= min(50.0, (expired_count / len(self.records)) * 100)
        
        # Deduct points for low consent rate
        consent_records = len([r for r in self.records if r.consent_given])
        personal_records = len([r for r in self.records if r.data_type == "personal_data"])
        
        if personal_records > 0:
            consent_rate = consent_records / personal_records
            if consent_rate < 0.8:  # Less than 80% consent
                score -= (0.8 - consent_rate) * 25  # Up to 25 points deduction
        
        return max(0.0, score)
    
    def _generate_compliance_recommendations(self, expired_records: List[ComplianceRecord]) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        if expired_records:
            recommendations.append(f"Delete or anonymize {len(expired_records)} expired data records")
        
        personal_records = [r for r in self.records if r.data_type == "personal_data"]
        consent_records = [r for r in personal_records if r.consent_given]
        
        if personal_records and len(consent_records) / len(personal_records) < 0.8:
            recommendations.append("Improve consent collection processes for personal data")
        
        if len(self.records) > 10000:
            recommendations.append("Consider implementing automated data retention policies")
        
        return recommendations
    
    def _create_data_hash(self, data: str) -> str:
        """Create hash signature for data integrity."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID."""
        timestamp = str(int(datetime.now().timestamp() * 1000000))
        return f"comp_{timestamp}"
    
    def _load_records(self):
        """Load existing compliance records."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                    self.records = [ComplianceRecord(**record) for record in data]
            except Exception as e:
                logger.error(f"Failed to load compliance records: {e}")
        
        if self.processing_records_file.exists():
            try:
                with open(self.processing_records_file, 'r') as f:
                    data = json.load(f)
                    self.processing_records = [DataProcessingRecord(**record) for record in data]
            except Exception as e:
                logger.error(f"Failed to load processing records: {e}")
    
    def _save_records(self):
        """Save compliance records."""
        try:
            with open(self.records_file, 'w') as f:
                json.dump([asdict(record) for record in self.records], f, indent=2)
            
            with open(self.processing_records_file, 'w') as f:
                json.dump([asdict(record) for record in self.processing_records], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save compliance records: {e}")


class SOC2Compliance:
    """SOC 2 compliance management."""
    
    def __init__(self):
        self.trust_service_criteria = {
            "security": ["CC6.1", "CC6.2", "CC6.3", "CC6.4", "CC6.5", "CC6.6", "CC6.7", "CC6.8"],
            "availability": ["A1.1", "A1.2", "A1.3"],
            "processing_integrity": ["PI1.1", "PI1.2", "PI1.3"],
            "confidentiality": ["C1.1", "C1.2"],
            "privacy": ["P1.1", "P2.1", "P3.1", "P3.2", "P4.1", "P4.2", "P4.3", "P5.1", "P5.2", "P6.1", "P6.2", "P6.3", "P7.1", "P8.1"]
        }
    
    def assess_soc2_readiness(self) -> Dict[str, Any]:
        """Assess SOC 2 readiness across trust service criteria."""
        assessment = {
            "overall_readiness": "partial",
            "criteria_assessment": {},
            "recommendations": [],
            "next_steps": []
        }
        
        for category, controls in self.trust_service_criteria.items():
            assessment["criteria_assessment"][category] = {
                "total_controls": len(controls),
                "implemented": 0,
                "partial": 0,
                "not_implemented": 0,
                "controls": {}
            }
        
        # This would be expanded with actual control assessments
        assessment["recommendations"] = [
            "Implement comprehensive logging and monitoring",
            "Establish formal incident response procedures",
            "Document security policies and procedures",
            "Conduct regular security assessments",
            "Implement access control and user management"
        ]
        
        return assessment


# Global compliance instances
gdpr_compliance = GDPRCompliance()
soc2_compliance = SOC2Compliance()