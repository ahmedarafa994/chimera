"""
Professional Security Report Generator Endpoints

Phase 2 feature for competitive differentiation:
- PDF/HTML reports suitable for client deliverables
- Executive summary with risk ratings
- Detailed findings with evidence and remediation
- Customizable branding options
"""

import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import base64

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.core.auth import get_current_user
from app.db.models import User, Assessment
from app.core.observability import get_logger

logger = get_logger("chimera.api.reports")
router = APIRouter()

# Report Models
class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ReportType(str, Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPREHENSIVE = "comprehensive"
    COMPLIANCE = "compliance"

class SecurityFinding(BaseModel):
    """Individual security finding in a report"""
    id: str
    title: str
    description: str
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)

    # Evidence
    affected_prompts: List[str] = Field(default_factory=list)
    model_responses: List[str] = Field(default_factory=list)
    technique_used: str

    # Impact assessment
    business_impact: str
    technical_impact: str
    exploitability: float = Field(ge=0.0, le=1.0)

    # Remediation
    remediation_steps: List[str] = Field(default_factory=list)
    prevention_measures: List[str] = Field(default_factory=list)
    priority: int = Field(ge=1, le=5)

class ExecutiveSummary(BaseModel):
    """Executive summary for reports"""
    overall_risk_score: float = Field(ge=0.0, le=10.0)
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int

    key_risks: List[str]
    business_recommendations: List[str]
    compliance_status: str
    assessment_scope: str
    tested_models: List[str]

class ReportBranding(BaseModel):
    """Customizable branding for reports"""
    company_name: Optional[str] = None
    company_logo: Optional[str] = None  # Base64 encoded
    report_title: Optional[str] = None
    prepared_for: Optional[str] = None
    prepared_by: Optional[str] = None
    contact_info: Optional[str] = None
    confidentiality_level: str = "CONFIDENTIAL"

class ReportRequest(BaseModel):
    """Request to generate a security report"""
    assessment_ids: List[str] = Field(min_items=1)
    report_type: ReportType = ReportType.TECHNICAL
    report_format: ReportFormat = ReportFormat.PDF
    include_executive_summary: bool = True
    include_detailed_findings: bool = True
    include_remediation_plan: bool = True
    include_appendix: bool = False

    # Customization
    branding: Optional[ReportBranding] = None
    custom_sections: Dict[str, str] = Field(default_factory=dict)

    # Filtering
    min_risk_level: Optional[RiskLevel] = None
    include_techniques: Optional[List[str]] = None
    exclude_techniques: Optional[List[str]] = None

class SecurityReport(BaseModel):
    """Complete security assessment report"""
    report_id: str
    report_type: ReportType
    report_format: ReportFormat
    generated_at: datetime

    # Metadata
    assessment_period: Dict[str, datetime]
    scope: Dict[str, Any]
    methodology: List[str]

    # Content
    executive_summary: ExecutiveSummary
    findings: List[SecurityFinding]
    recommendations: List[str]

    # Statistics
    statistics: Dict[str, Any]

    # Branding
    branding: Optional[ReportBranding] = None

class ReportListResponse(BaseModel):
    """List of generated reports"""
    reports: List[Dict[str, Any]]
    total: int

@router.post("/generate", response_model=Dict[str, str])
async def generate_security_report(
    report_request: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a professional security assessment report

    This endpoint creates comprehensive security reports suitable for
    client deliverables, executive presentations, and compliance documentation.
    """
    try:
        # Generate report ID
        report_id = f"rpt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Generating security report {report_id} for user {current_user.id}")

        # Validate and fetch assessments
        assessments = []
        for assessment_id in report_request.assessment_ids:
            assessment = db.query(Assessment).filter(
                Assessment.id == assessment_id,
                Assessment.user_id == current_user.id
            ).first()

            if not assessment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Assessment {assessment_id} not found"
                )

            if assessment.status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Assessment {assessment_id} is not completed"
                )

            assessments.append(assessment)

        if not assessments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid completed assessments found"
            )

        # Generate report content
        report = await generate_report_content(
            assessments, report_request, report_id, current_user
        )

        # Generate the actual report file
        report_file = await generate_report_file(report, report_request.report_format)

        # Store report metadata (in production, would save to database)
        # For now, we'll return the report ID and format info

        logger.info(f"Generated security report {report_id} ({report_request.report_format.value})")

        return {
            "report_id": report_id,
            "format": report_request.report_format.value,
            "status": "completed",
            "download_url": f"/api/v1/reports/{report_id}/download",
            "size_bytes": len(report_file) if report_file else 0,
            "generated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate security report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate security report"
        )

async def generate_report_content(
    assessments: List[Assessment],
    request: ReportRequest,
    report_id: str,
    user: User
) -> SecurityReport:
    """Generate the structured content for the security report"""

    # Analyze assessments to extract findings
    findings = []
    all_techniques = []
    all_models = set()

    for assessment in assessments:
        all_models.add(f"{assessment.target_provider}/{assessment.target_model}")
        all_techniques.extend(assessment.technique_ids)

        # Convert assessment results to findings
        if assessment.results:
            for result in assessment.results:
                if isinstance(result, dict):
                    finding = create_finding_from_result(assessment, result)
                    if finding and (not request.min_risk_level or
                                  get_risk_level_score(finding.risk_level) >= get_risk_level_score(request.min_risk_level)):
                        findings.append(finding)

    # Filter findings based on request
    if request.include_techniques:
        findings = [f for f in findings if f.technique_used in request.include_techniques]

    if request.exclude_techniques:
        findings = [f for f in findings if f.technique_used not in request.exclude_techniques]

    # Sort findings by risk level and priority
    findings.sort(key=lambda x: (get_risk_level_score(x.risk_level), x.priority), reverse=True)

    # Generate executive summary
    exec_summary = generate_executive_summary(findings, assessments, list(all_models))

    # Calculate assessment period
    start_dates = [a.started_at or a.created_at for a in assessments if a.started_at or a.created_at]
    end_dates = [a.completed_at or a.updated_at for a in assessments if a.completed_at or a.updated_at]

    assessment_period = {
        "start": min(start_dates) if start_dates else datetime.utcnow(),
        "end": max(end_dates) if end_dates else datetime.utcnow()
    }

    # Generate methodology list
    methodology = list(set(all_techniques))

    # Generate statistics
    statistics = generate_report_statistics(assessments, findings)

    # Generate recommendations
    recommendations = generate_recommendations(findings)

    return SecurityReport(
        report_id=report_id,
        report_type=request.report_type,
        report_format=request.report_format,
        generated_at=datetime.utcnow(),
        assessment_period=assessment_period,
        scope={
            "assessments_analyzed": len(assessments),
            "models_tested": list(all_models),
            "techniques_used": len(set(all_techniques)),
            "test_duration_hours": (assessment_period["end"] - assessment_period["start"]).total_seconds() / 3600
        },
        methodology=methodology,
        executive_summary=exec_summary,
        findings=findings,
        recommendations=recommendations,
        statistics=statistics,
        branding=request.branding
    )

def create_finding_from_result(assessment: Assessment, result: Dict[str, Any]) -> Optional[SecurityFinding]:
    """Create a security finding from an assessment result"""

    # Determine if this result represents a security issue
    if not result.get("vulnerability_detected", False) and not result.get("attack_success", False):
        return None

    # Determine risk level based on success and context
    risk_level = determine_risk_level(result, assessment)

    finding_id = f"finding_{assessment.id}_{result.get('provider', 'unknown')}"

    return SecurityFinding(
        id=finding_id,
        title=f"Prompt Injection Vulnerability - {result.get('provider', 'Unknown Provider')}",
        description=f"The {result.get('provider')} model demonstrated vulnerability to {assessment.technique_ids[0] if assessment.technique_ids else 'attack techniques'}.",
        risk_level=risk_level,
        confidence=result.get("safety_score", 0.0),
        affected_prompts=[assessment.original_prompt],
        model_responses=[result.get("response_text", "")[:500] + "..." if result.get("response_text") else ""],
        technique_used=assessment.technique_ids[0] if assessment.technique_ids else "unknown",
        business_impact="Potential for inappropriate content generation and policy violations",
        technical_impact="Model safety controls can be bypassed using adversarial prompts",
        exploitability=1.0 - (result.get("safety_score", 0.0)),
        remediation_steps=[
            "Implement additional input validation and filtering",
            "Update model safety guidelines and training",
            "Deploy prompt injection detection systems",
            "Establish content moderation workflows"
        ],
        prevention_measures=[
            "Regular security testing of AI systems",
            "Implement defense-in-depth strategies",
            "Monitor for unusual prompt patterns",
            "Maintain updated safety policies"
        ],
        priority=get_risk_priority(risk_level)
    )

def determine_risk_level(result: Dict[str, Any], assessment: Assessment) -> RiskLevel:
    """Determine the risk level for a finding"""
    safety_score = result.get("safety_score", 1.0)
    attack_success = result.get("attack_success", False)

    if safety_score < 0.2 and attack_success:
        return RiskLevel.CRITICAL
    elif safety_score < 0.4 and attack_success:
        return RiskLevel.HIGH
    elif safety_score < 0.6:
        return RiskLevel.MEDIUM
    elif safety_score < 0.8:
        return RiskLevel.LOW
    else:
        return RiskLevel.INFO

def get_risk_level_score(risk_level: RiskLevel) -> int:
    """Get numeric score for risk level sorting"""
    scores = {
        RiskLevel.CRITICAL: 5,
        RiskLevel.HIGH: 4,
        RiskLevel.MEDIUM: 3,
        RiskLevel.LOW: 2,
        RiskLevel.INFO: 1
    }
    return scores[risk_level]

def get_risk_priority(risk_level: RiskLevel) -> int:
    """Get priority number for risk level"""
    priorities = {
        RiskLevel.CRITICAL: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.MEDIUM: 3,
        RiskLevel.LOW: 4,
        RiskLevel.INFO: 5
    }
    return priorities[risk_level]

def generate_executive_summary(
    findings: List[SecurityFinding],
    assessments: List[Assessment],
    models: List[str]
) -> ExecutiveSummary:
    """Generate executive summary from findings"""

    # Count findings by risk level
    risk_counts = {level: 0 for level in RiskLevel}
    for finding in findings:
        risk_counts[finding.risk_level] += 1

    # Calculate overall risk score (0-10)
    risk_weights = {RiskLevel.CRITICAL: 10, RiskLevel.HIGH: 7, RiskLevel.MEDIUM: 4, RiskLevel.LOW: 2, RiskLevel.INFO: 1}
    total_weight = sum(risk_counts[level] * risk_weights[level] for level in RiskLevel)
    max_possible = len(findings) * 10 if findings else 1
    overall_risk_score = (total_weight / max_possible) * 10

    # Generate key risks
    key_risks = []
    for finding in findings[:3]:  # Top 3 findings
        key_risks.append(f"{finding.title} ({finding.risk_level.value.upper()})")

    # Generate business recommendations
    business_recommendations = [
        "Implement comprehensive AI security testing program",
        "Establish clear AI usage policies and guidelines",
        "Deploy prompt injection detection and prevention systems",
        "Conduct regular security assessments of AI systems"
    ]

    if risk_counts[RiskLevel.CRITICAL] > 0:
        business_recommendations.insert(0, "Address critical vulnerabilities immediately")

    return ExecutiveSummary(
        overall_risk_score=overall_risk_score,
        total_findings=len(findings),
        critical_findings=risk_counts[RiskLevel.CRITICAL],
        high_findings=risk_counts[RiskLevel.HIGH],
        medium_findings=risk_counts[RiskLevel.MEDIUM],
        low_findings=risk_counts[RiskLevel.LOW],
        key_risks=key_risks,
        business_recommendations=business_recommendations,
        compliance_status="Requires Attention" if risk_counts[RiskLevel.CRITICAL] + risk_counts[RiskLevel.HIGH] > 0 else "Good",
        assessment_scope=f"{len(assessments)} security assessments across {len(models)} AI models",
        tested_models=models
    )

def generate_report_statistics(assessments: List[Assessment], findings: List[SecurityFinding]) -> Dict[str, Any]:
    """Generate statistics section for the report"""

    total_tests = len(assessments)
    successful_attacks = len(findings)

    # Technique effectiveness
    technique_stats = {}
    for assessment in assessments:
        for technique in assessment.technique_ids:
            if technique not in technique_stats:
                technique_stats[technique] = {"total": 0, "successful": 0}
            technique_stats[technique]["total"] += 1

            # Check if this assessment had successful attacks
            if assessment.results and any(
                result.get("vulnerability_detected") or result.get("attack_success")
                for result in assessment.results if isinstance(result, dict)
            ):
                technique_stats[technique]["successful"] += 1

    # Calculate success rates
    technique_success_rates = {
        technique: (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        for technique, stats in technique_stats.items()
    }

    return {
        "total_assessments": total_tests,
        "successful_attacks": successful_attacks,
        "overall_success_rate": (successful_attacks / total_tests) * 100 if total_tests > 0 else 0,
        "technique_success_rates": technique_success_rates,
        "finding_distribution": {
            level.value: len([f for f in findings if f.risk_level == level])
            for level in RiskLevel
        },
        "most_effective_technique": max(technique_success_rates.items(), key=lambda x: x[1])[0] if technique_success_rates else None,
        "average_confidence": sum(f.confidence for f in findings) / len(findings) if findings else 0
    }

def generate_recommendations(findings: List[SecurityFinding]) -> List[str]:
    """Generate high-level recommendations from findings"""

    recommendations = []

    critical_high = [f for f in findings if f.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]]

    if critical_high:
        recommendations.append("Immediately address critical and high-risk vulnerabilities identified in this assessment")
        recommendations.append("Implement emergency response procedures for prompt injection attacks")

    if len(findings) > 5:
        recommendations.append("Develop a comprehensive AI security strategy given the multiple vulnerabilities found")

    recommendations.extend([
        "Establish continuous monitoring for prompt injection attempts",
        "Implement defense-in-depth security controls for AI systems",
        "Develop incident response procedures for AI security breaches",
        "Conduct regular security training for teams working with AI systems",
        "Consider implementing zero-trust architecture for AI deployments"
    ])

    return recommendations

async def generate_report_file(report: SecurityReport, format: ReportFormat) -> bytes:
    """Generate the actual report file in the specified format"""

    if format == ReportFormat.JSON:
        return json.dumps(report.model_dump(), indent=2, default=str).encode()

    elif format == ReportFormat.HTML:
        return generate_html_report(report)

    elif format == ReportFormat.PDF:
        # For PDF generation, we'd typically use a library like ReportLab or WeasyPrint
        # For now, we'll generate HTML and indicate it needs PDF conversion
        html_content = generate_html_report(report)
        # In production: return convert_html_to_pdf(html_content)
        return html_content  # Placeholder

    else:
        raise ValueError(f"Unsupported report format: {format}")

def generate_html_report(report: SecurityReport) -> bytes:
    """Generate HTML report content"""

    # Risk level colors
    risk_colors = {
        RiskLevel.CRITICAL: "#dc2626",
        RiskLevel.HIGH: "#ea580c",
        RiskLevel.MEDIUM: "#d97706",
        RiskLevel.LOW: "#65a30d",
        RiskLevel.INFO: "#6b7280"
    }

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Security Assessment Report - {report.report_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #e5e7eb; padding-bottom: 20px; }}
            .section {{ margin: 30px 0; }}
            .risk-critical {{ color: {risk_colors[RiskLevel.CRITICAL]}; }}
            .risk-high {{ color: {risk_colors[RiskLevel.HIGH]}; }}
            .risk-medium {{ color: {risk_colors[RiskLevel.MEDIUM]}; }}
            .risk-low {{ color: {risk_colors[RiskLevel.LOW]}; }}
            .risk-info {{ color: {risk_colors[RiskLevel.INFO]}; }}
            .finding {{ border: 1px solid #e5e7eb; margin: 20px 0; padding: 20px; border-radius: 8px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
            .metric-card {{ background: #f9fafb; padding: 20px; border-radius: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Security Assessment Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.generated_at.strftime('%B %d, %Y at %I:%M %p UTC')}</p>
                {f"<p><strong>Prepared for:</strong> {report.branding.prepared_for}</p>" if report.branding and report.branding.prepared_for else ""}
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Overall Risk Score</h3>
                        <div style="font-size: 2em; font-weight: bold; color: {risk_colors[RiskLevel.HIGH if report.executive_summary.overall_risk_score > 7 else RiskLevel.MEDIUM if report.executive_summary.overall_risk_score > 4 else RiskLevel.LOW]};">
                            {report.executive_summary.overall_risk_score:.1f}/10
                        </div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Findings</h3>
                        <div style="font-size: 2em; font-weight: bold;">{report.executive_summary.total_findings}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Critical Issues</h3>
                        <div style="font-size: 2em; font-weight: bold; color: {risk_colors[RiskLevel.CRITICAL]};">{report.executive_summary.critical_findings}</div>
                    </div>
                    <div class="metric-card">
                        <h3>High Risk Issues</h3>
                        <div style="font-size: 2em; font-weight: bold; color: {risk_colors[RiskLevel.HIGH]};">{report.executive_summary.high_findings}</div>
                    </div>
                </div>

                <h3>Key Risks</h3>
                <ul>
                    {''.join(f'<li>{risk}</li>' for risk in report.executive_summary.key_risks)}
                </ul>

                <h3>Business Recommendations</h3>
                <ol>
                    {''.join(f'<li>{rec}</li>' for rec in report.executive_summary.business_recommendations)}
                </ol>
            </div>

            <div class="section">
                <h2>Assessment Scope</h2>
                <p><strong>Assessment Period:</strong> {report.assessment_period['start'].strftime('%B %d, %Y')} - {report.assessment_period['end'].strftime('%B %d, %Y')}</p>
                <p><strong>Models Tested:</strong> {', '.join(report.executive_summary.tested_models)}</p>
                <p><strong>Techniques Used:</strong> {', '.join(report.methodology)}</p>
                <p><strong>Total Assessments:</strong> {report.scope['assessments_analyzed']}</p>
            </div>

            <div class="section">
                <h2>Detailed Findings</h2>
                {''.join(generate_finding_html(finding, risk_colors) for finding in report.findings)}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ol>
                    {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
                </ol>
            </div>

            <div class="section">
                <h2>Statistics</h2>
                <p><strong>Overall Success Rate:</strong> {report.statistics.get('overall_success_rate', 0):.1f}%</p>
                <p><strong>Most Effective Technique:</strong> {report.statistics.get('most_effective_technique', 'N/A')}</p>
                <p><strong>Average Confidence:</strong> {report.statistics.get('average_confidence', 0):.2f}</p>
            </div>

            <div style="margin-top: 40px; text-align: center; color: #6b7280; font-size: 0.9em;">
                <p>This report was generated by Chimera AI Security Platform</p>
                <p>Report ID: {report.report_id} | Generated: {report.generated_at.isoformat()}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content.encode()

def generate_finding_html(finding: SecurityFinding, risk_colors: Dict[RiskLevel, str]) -> str:
    """Generate HTML for a single finding"""

    risk_class = f"risk-{finding.risk_level.value}"

    return f"""
    <div class="finding">
        <h3 class="{risk_class}">{finding.title} <span style="font-size: 0.8em;">({finding.risk_level.value.upper()})</span></h3>
        <p><strong>Description:</strong> {finding.description}</p>
        <p><strong>Technique Used:</strong> {finding.technique_used}</p>
        <p><strong>Confidence:</strong> {finding.confidence:.2f}</p>
        <p><strong>Exploitability:</strong> {finding.exploitability:.2f}</p>

        <h4>Business Impact</h4>
        <p>{finding.business_impact}</p>

        <h4>Technical Impact</h4>
        <p>{finding.technical_impact}</p>

        <h4>Remediation Steps</h4>
        <ol>
            {''.join(f'<li>{step}</li>' for step in finding.remediation_steps)}
        </ol>

        <h4>Prevention Measures</h4>
        <ul>
            {''.join(f'<li>{measure}</li>' for measure in finding.prevention_measures)}
        </ul>
    </div>
    """

@router.get("/", response_model=ReportListResponse)
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List generated security reports for the current user"""
    try:
        # In production, this would query a reports table
        # For now, return empty list as placeholder
        return ReportListResponse(reports=[], total=0)

    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve reports"
        )

@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download a generated report file"""
    try:
        # In production, this would fetch the report from storage
        # For now, return a placeholder response

        # Determine format from report ID or separate parameter
        content = b"Sample report content - not yet implemented"

        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=security_report_{report_id}.pdf"
            }
        )

    except Exception as e:
        logger.error(f"Failed to download report {report_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report"
        )