"""Reflection prompt templates and utilities."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PromptDomain(Enum):
    """Domains for specialized reflection prompts."""
    GENERAL = "general"
    SOFTWARE_ENGINEERING = "software_engineering"  
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    PROBLEM_SOLVING = "problem_solving"


@dataclass
class ReflectionPromptSet:
    """Set of prompts for a specific reflection scenario."""
    initial_reflection: str
    improvement_strategy: str
    success_evaluation: str
    failure_analysis: str


class ReflectionPrompts:
    """Collection of reflection prompt templates for different domains."""
    
    GENERAL_PROMPTS = ReflectionPromptSet(
        initial_reflection="""
Analyze your previous response and identify any issues:

1. **Accuracy**: Is the information correct and factual?
2. **Completeness**: Did you address all parts of the request?
3. **Clarity**: Is your response clear and easy to understand?
4. **Relevance**: Does your response directly address what was asked?

List specific issues you identified:
""",
        
        improvement_strategy="""
Based on the issues identified, create an improvement strategy:

1. **Priority Issues**: What are the most critical problems to fix?
2. **Approach**: How will you address each issue?
3. **Verification**: How will you ensure improvements work?
4. **Prevention**: How will you avoid similar issues in the future?

Provide specific action steps:
""",
        
        success_evaluation="""
Evaluate if this attempt was successful:

1. **Requirements Met**: Were all original requirements satisfied?
2. **Quality**: Is the quality acceptable for the intended use?
3. **Completeness**: Is anything missing that should be included?
4. **User Satisfaction**: Would this meet user expectations?

Rate success (0.0-1.0) and explain:
""",
        
        failure_analysis="""
Analyze what went wrong in this attempt:

1. **Root Cause**: What was the fundamental cause of failure?
2. **Contributing Factors**: What other factors made it worse?
3. **Decision Points**: Where could different choices have helped?
4. **Learning**: What key lessons can be extracted?

Provide actionable insights:
"""
    )
    
    SOFTWARE_ENGINEERING_PROMPTS = ReflectionPromptSet(
        initial_reflection="""
Review your code solution and identify potential issues:

1. **Correctness**: Does the code solve the problem correctly?
2. **Edge Cases**: Are edge cases and error conditions handled?
3. **Performance**: Is the solution efficient for the expected use case?
4. **Security**: Are there any security vulnerabilities?
5. **Maintainability**: Is the code readable and well-structured?
6. **Testing**: Can this code be easily tested?

List specific code issues:
""",
        
        improvement_strategy="""
Create a code improvement plan:

1. **Bug Fixes**: What bugs need to be fixed first?
2. **Refactoring**: What code structure improvements are needed?
3. **Optimization**: Where can performance be improved?
4. **Security**: What security measures should be added?
5. **Documentation**: What documentation is needed?

Prioritize improvements:
""",
        
        success_evaluation="""
Evaluate the code quality:

1. **Functionality**: Does it work as intended?
2. **Test Coverage**: Would it pass comprehensive tests?
3. **Code Standards**: Does it follow best practices?
4. **Production Ready**: Could this be deployed safely?

Rate code quality (0.0-1.0):
""",
        
        failure_analysis="""
Analyze code failure patterns:

1. **Syntax Errors**: Are there syntax or compilation issues?
2. **Logic Errors**: Where is the logic flawed?
3. **Architecture**: Are there design problems?
4. **Requirements**: Did you misunderstand requirements?

Identify root causes:
"""
    )
    
    DATA_ANALYSIS_PROMPTS = ReflectionPromptSet(
        initial_reflection="""
Review your data analysis approach:

1. **Data Quality**: Did you properly validate and clean the data?
2. **Methodology**: Are your analytical methods appropriate?
3. **Assumptions**: What assumptions did you make and are they valid?
4. **Bias**: Could there be bias in your analysis or conclusions?
5. **Statistical Validity**: Are your statistical methods sound?

Identify analytical issues:
""",
        
        improvement_strategy="""
Plan analytical improvements:

1. **Data Validation**: How can you better validate data quality?
2. **Method Selection**: Should you use different analytical methods?
3. **Bias Mitigation**: How can you reduce potential bias?
4. **Validation**: How can you validate your findings?

Outline improvement steps:
""",
        
        success_evaluation="""
Evaluate analysis quality:

1. **Accuracy**: Are your findings likely to be accurate?
2. **Significance**: Are the results statistically significant?
3. **Actionability**: Can others act on these insights?
4. **Reproducibility**: Could others reproduce your analysis?

Rate analysis quality (0.0-1.0):
""",
        
        failure_analysis="""
Analyze analytical failures:

1. **Data Issues**: Were there problems with data quality or access?
2. **Method Problems**: Were inappropriate methods used?
3. **Interpretation**: Were results misinterpreted?
4. **Communication**: Were findings poorly communicated?

Identify core issues:
"""
    )
    
    CREATIVE_WRITING_PROMPTS = ReflectionPromptSet(
        initial_reflection="""
Evaluate your creative writing:

1. **Engagement**: Is the content engaging and interesting?
2. **Clarity**: Is the message clear and well-communicated?
3. **Style**: Is the writing style appropriate for the audience?
4. **Structure**: Is the content well-organized and flows logically?
5. **Originality**: Is there creative and original thinking?

Assess creative elements:
""",
        
        improvement_strategy="""
Plan creative improvements:

1. **Content**: What content improvements would help?
2. **Style**: How can writing style be enhanced?
3. **Structure**: What organizational improvements are needed?
4. **Voice**: How can you develop a stronger voice?

Define enhancement approach:
""",
        
        success_evaluation="""
Rate creative success:

1. **Impact**: Does it have the intended emotional/intellectual impact?
2. **Audience Fit**: Is it appropriate for the target audience?
3. **Quality**: Is the overall quality professional/polished?
4. **Goals**: Does it meet the original creative goals?

Evaluate creative success (0.0-1.0):
""",
        
        failure_analysis="""
Analyze creative shortcomings:

1. **Message**: Was the core message unclear or weak?
2. **Execution**: Was the execution poor despite good ideas?
3. **Audience**: Did you misjudge the audience needs?
4. **Process**: Were there problems in the creative process?

Identify creative blocks:
"""
    )

    @classmethod
    def for_domain(cls, domain: PromptDomain) -> ReflectionPromptSet:
        """Get prompt set for a specific domain."""
        domain_map = {
            PromptDomain.GENERAL: cls.GENERAL_PROMPTS,
            PromptDomain.SOFTWARE_ENGINEERING: cls.SOFTWARE_ENGINEERING_PROMPTS,
            PromptDomain.DATA_ANALYSIS: cls.DATA_ANALYSIS_PROMPTS,
            PromptDomain.CREATIVE_WRITING: cls.CREATIVE_WRITING_PROMPTS,
            PromptDomain.RESEARCH: cls.GENERAL_PROMPTS,  # Use general for now
            PromptDomain.PROBLEM_SOLVING: cls.GENERAL_PROMPTS  # Use general for now
        }
        return domain_map.get(domain, cls.GENERAL_PROMPTS)
    
    @classmethod
    def build_reflection_prompt(
        cls, 
        domain: PromptDomain, 
        task: str, 
        output: str, 
        issues: List[str],
        iteration: int = 0
    ) -> str:
        """Build a comprehensive reflection prompt."""
        prompt_set = cls.for_domain(domain)
        
        context = f"""
TASK: {task}

CURRENT OUTPUT: {output}

ITERATION: {iteration + 1}
"""
        
        if issues:
            context += f"""
PREVIOUS ISSUES IDENTIFIED:
{chr(10).join(f"- {issue}" for issue in issues)}
"""
        
        reflection_prompt = f"{context}\n\n{prompt_set.initial_reflection}"
        
        return reflection_prompt
    
    @classmethod  
    def build_improvement_prompt(
        cls,
        domain: PromptDomain,
        task: str,
        issues: List[str],
        improvements: List[str]
    ) -> str:
        """Build an improvement strategy prompt."""
        prompt_set = cls.for_domain(domain)
        
        context = f"""
ORIGINAL TASK: {task}

IDENTIFIED ISSUES:
{chr(10).join(f"- {issue}" for issue in issues)}

SUGGESTED IMPROVEMENTS:
{chr(10).join(f"- {improvement}" for improvement in improvements)}
"""
        
        return f"{context}\n\n{prompt_set.improvement_strategy}"


class CustomReflectionPrompts:
    """Utility for creating custom reflection prompts."""
    
    @staticmethod
    def create_domain_specific_prompt(
        domain_description: str,
        evaluation_criteria: List[str],
        common_failure_modes: List[str]
    ) -> ReflectionPromptSet:
        """Create custom prompts for a specific domain."""
        
        criteria_text = "\n".join(f"{i+1}. **{criterion}**" for i, criterion in enumerate(evaluation_criteria))
        failure_modes_text = "\n".join(f"- {mode}" for mode in common_failure_modes)
        
        return ReflectionPromptSet(
            initial_reflection=f"""
Analyze your response in the context of {domain_description}:

{criteria_text}

Common failure modes to check for:
{failure_modes_text}

Identify specific issues:
""",
            
            improvement_strategy=f"""
Based on {domain_description} best practices, create an improvement strategy:

1. **Priority Issues**: What are the most critical problems for {domain_description}?
2. **Domain-Specific Approach**: How should you address issues given {domain_description} constraints?
3. **Validation**: How will you verify improvements work in {domain_description}?

Provide domain-appropriate action steps:
""",
            
            success_evaluation=f"""
Evaluate success in {domain_description} context:

{criteria_text}

Rate overall success for {domain_description} (0.0-1.0) and explain:
""",
            
            failure_analysis=f"""
Analyze failures specific to {domain_description}:

1. **Domain Constraints**: What {domain_description} constraints were violated?
2. **Best Practices**: What {domain_description} best practices were ignored?
3. **Context**: What context was missed for {domain_description}?

Provide domain-specific insights:
"""
        )