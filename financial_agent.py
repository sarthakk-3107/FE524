"""
Financial Analysis Agent - Autonomous multi-step reasoning
FE524 Project - Agentic RAG Enhancement

This agent can:
1. Break down complex financial queries into sub-tasks
2. Execute retrieval and analysis steps autonomously
3. Perform comparative analysis across multiple documents
4. Generate comprehensive financial reports
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI


class AgentAction(Enum):
    """Available actions the agent can take"""
    RETRIEVE_INFO = "retrieve_info"
    CALCULATE_METRIC = "calculate_metric"
    COMPARE_VALUES = "compare_values"
    SYNTHESIZE_ANSWER = "synthesize_answer"
    SEARCH_SPECIFIC_SECTION = "search_specific_section"
    FINISH = "finish"


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning process"""
    step_number: int
    action: AgentAction
    reasoning: str
    query: str
    result: Optional[Any] = None
    observations: str = ""


class FinancialAgent:
    """Autonomous agent for complex financial analysis tasks"""

    def __init__(self, rag_system, openai_client: OpenAI):
        self.rag_system = rag_system
        self.client = openai_client
        self.model = "gpt-4o-mini"
        self.max_steps = 10
        self.thought_history: List[AgentStep] = []

    def reset(self):
        """Reset agent state for new query"""
        self.thought_history = []

    def plan_analysis(self, user_query: str) -> List[Dict[str, str]]:
        """Create a step-by-step plan to answer the user's query"""

        planning_prompt = f"""You are a financial analysis agent. Break down this complex query into specific, actionable steps.

User Query: {user_query}

Available Actions:
1. RETRIEVE_INFO: Search the 10-K for specific information (e.g., "find revenue data")
2. CALCULATE_METRIC: Compute financial ratios or metrics (e.g., "calculate profit margin")
3. COMPARE_VALUES: Compare metrics across periods or categories (e.g., "compare Q1 vs Q2 revenue")
4. SEARCH_SPECIFIC_SECTION: Look in a specific 10-K section (e.g., "search risk factors section")
5. SYNTHESIZE_ANSWER: Combine findings to create final answer

Create a plan with 3-6 steps. Format as JSON array:
[
  {{"step": 1, "action": "RETRIEVE_INFO", "reasoning": "why this step", "query": "specific search query"}},
  {{"step": 2, "action": "CALCULATE_METRIC", "reasoning": "why calculate", "query": "what to calculate"}},
  ...
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial planning expert. Output only valid JSON."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.3
            )

            plan_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if plan_text.startswith("```"):
                plan_text = plan_text.split("```")[1]
                if plan_text.startswith("json"):
                    plan_text = plan_text[4:]

            plan = json.loads(plan_text)
            return plan
        except Exception as e:
            # Fallback: simple single-step plan
            return [{
                "step": 1,
                "action": "RETRIEVE_INFO",
                "reasoning": "Retrieve relevant information to answer query",
                "query": user_query
            }]

    def execute_retrieve_info(self, query: str) -> Dict[str, Any]:
        """Execute information retrieval action"""
        if not self.rag_system.chunks:
            return {"error": "No documents loaded"}

        retrieved = self.rag_system.hybrid_retrieve(query, top_k=5)

        # Summarize findings
        context = "\n\n".join([chunk['text'][:500] for chunk in retrieved[:3]])

        return {
            "chunks": retrieved,
            "summary_context": context,
            "num_sources": len(retrieved)
        }

    def execute_calculate_metric(self, calculation_query: str, context: str) -> Dict[str, Any]:
        """Execute metric calculation using LLM"""

        calc_prompt = f"""Based on this financial data, perform the requested calculation.

Context:
{context}

Calculation Request: {calculation_query}

Provide:
1. Numbers extracted from context
2. Formula used
3. Step-by-step calculation
4. Final result

Format as JSON:
{{
    "extracted_numbers": {{"item": "value"}},
    "formula": "formula description",
    "steps": ["step 1", "step 2"],
    "result": {{"metric_name": "value"}},
    "confidence": "high/medium/low"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial calculator. Output valid JSON only."},
                    {"role": "user", "content": calc_prompt}
                ],
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            return json.loads(result_text)
        except:
            return {
                "error": "Could not parse calculation",
                "result": {},
                "confidence": "low"
            }

    def execute_compare_values(self, comparison_query: str, context: str) -> Dict[str, Any]:
        """Execute comparison analysis"""

        compare_prompt = f"""Compare the requested values from this financial data.

Context:
{context}

Comparison Request: {comparison_query}

Provide comparison analysis as JSON:
{{
    "values_compared": {{"item1": "value1", "item2": "value2"}},
    "difference": "absolute difference",
    "percent_change": "percentage change",
    "trend": "increasing/decreasing/stable",
    "interpretation": "what this means"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Output valid JSON only."},
                    {"role": "user", "content": compare_prompt}
                ],
                temperature=0.2
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            return json.loads(result_text)
        except:
            return {"error": "Could not perform comparison"}

    def execute_search_section(self, section_query: str) -> Dict[str, Any]:
        """Search a specific section of the 10-K"""

        # Modify query to target specific section
        section_keywords = {
            "risk": "Item 1A Risk Factors",
            "business": "Item 1 Business",
            "md&a": "Item 7 Management Discussion",
            "financial statements": "Item 8 Financial Statements",
            "governance": "Item 10 Corporate Governance"
        }

        # Enhance query with section context
        enhanced_query = section_query
        for keyword, section_name in section_keywords.items():
            if keyword in section_query.lower():
                enhanced_query = f"{section_name}: {section_query}"
                break

        return self.execute_retrieve_info(enhanced_query)

    def execute_step(self, step: Dict[str, Any]) -> AgentStep:
        """Execute a single step in the agent's plan"""

        action_str = step.get("action", "RETRIEVE_INFO")
        try:
            action = AgentAction[action_str]
        except:
            action = AgentAction.RETRIEVE_INFO

        agent_step = AgentStep(
            step_number=step.get("step", len(self.thought_history) + 1),
            action=action,
            reasoning=step.get("reasoning", ""),
            query=step.get("query", "")
        )

        # Execute based on action type
        if action == AgentAction.RETRIEVE_INFO:
            result = self.execute_retrieve_info(agent_step.query)
            agent_step.result = result
            agent_step.observations = f"Retrieved {result.get('num_sources', 0)} relevant sources"

        elif action == AgentAction.SEARCH_SPECIFIC_SECTION:
            result = self.execute_search_section(agent_step.query)
            agent_step.result = result
            agent_step.observations = f"Searched specific section, found {result.get('num_sources', 0)} results"

        elif action == AgentAction.CALCULATE_METRIC:
            # Get context from previous steps
            context = self._get_accumulated_context()
            result = self.execute_calculate_metric(agent_step.query, context)
            agent_step.result = result
            agent_step.observations = f"Calculated: {list(result.get('result', {}).keys())}"

        elif action == AgentAction.COMPARE_VALUES:
            context = self._get_accumulated_context()
            result = self.execute_compare_values(agent_step.query, context)
            agent_step.result = result
            agent_step.observations = f"Comparison: {result.get('trend', 'unknown')} trend"

        self.thought_history.append(agent_step)
        return agent_step

    def _get_accumulated_context(self) -> str:
        """Get all context accumulated from previous steps"""
        contexts = []
        for step in self.thought_history:
            if step.result and isinstance(step.result, dict):
                if 'summary_context' in step.result:
                    contexts.append(step.result['summary_context'])
                elif 'chunks' in step.result:
                    for chunk in step.result['chunks'][:2]:
                        contexts.append(chunk['text'][:500])

        return "\n\n".join(contexts[:5])  # Limit context size

    def synthesize_final_answer(self, user_query: str) -> str:
        """Synthesize final answer from all steps"""

        # Compile all findings
        findings = []
        for step in self.thought_history:
            findings.append({
                "action": step.action.value,
                "query": step.query,
                "observations": step.observations,
                "result_summary": str(step.result)[:500] if step.result else "No result"
            })

        synthesis_prompt = f"""You are a financial analyst. Synthesize a comprehensive answer from these analysis steps.

Original Question: {user_query}

Analysis Steps Completed:
{json.dumps(findings, indent=2)}

Full Context from Documents:
{self._get_accumulated_context()}

Provide a clear, professional answer that:
1. Directly answers the original question
2. Cites specific findings from the analysis
3. Includes calculated metrics where applicable
4. Provides context and interpretation
5. Is formatted in clear paragraphs (not bullet points unless data tables)

Write a comprehensive but concise answer (200-400 words)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst providing comprehensive insights."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.4
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error synthesizing answer: {str(e)}"

    def run(self, user_query: str) -> Dict[str, Any]:
        """Main agent execution loop"""

        self.reset()

        # Step 1: Create plan
        plan = self.plan_analysis(user_query)

        # Step 2: Execute each step in plan
        for step_spec in plan[:self.max_steps]:
            step_result = self.execute_step(step_spec)

            # Early stopping if error
            if step_result.result and isinstance(step_result.result, dict):
                if 'error' in step_result.result:
                    break

        # Step 3: Synthesize final answer
        final_answer = self.synthesize_final_answer(user_query)

        # Compile response
        return {
            "answer": final_answer,
            "plan": plan,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action.value,
                    "reasoning": s.reasoning,
                    "query": s.query,
                    "observations": s.observations
                }
                for s in self.thought_history
            ],
            "num_steps": len(self.thought_history),
            "sources": self._collect_all_sources()
        }

    def _collect_all_sources(self) -> List[Dict]:
        """Collect all sources from all retrieval steps"""
        all_sources = []
        seen_texts = set()

        for step in self.thought_history:
            if step.result and isinstance(step.result, dict) and 'chunks' in step.result:
                for chunk in step.result['chunks']:
                    text_hash = hash(chunk['text'][:100])
                    if text_hash not in seen_texts:
                        all_sources.append(chunk)
                        seen_texts.add(text_hash)

        return all_sources[:10]  # Limit to top 10 sources
