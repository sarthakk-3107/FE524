"""
Financial Visualization Agent - Autonomous chart generation
FE524 Project - Visual Analytics Enhancement

This agent can:
1. Extract time-series financial data from 10-K filings
2. Identify trends and patterns
3. Generate interactive charts (revenue, expenses, margins, etc.)
4. Create comparative visualizations
5. Provide visual insights alongside narrative analysis
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI


class ChartType(Enum):
    """Types of charts the agent can generate"""
    LINE = "line"  # Trends over time
    BAR = "bar"  # Comparisons
    WATERFALL = "waterfall"  # Changes breakdown
    PIE = "pie"  # Composition
    SCATTER = "scatter"  # Correlations
    AREA = "area"  # Cumulative trends


@dataclass
class FinancialDataPoint:
    """Single financial data point"""
    metric: str
    value: float
    period: str
    unit: str  # e.g., "millions", "billions"


@dataclass
class VisualizationResult:
    """Result from visualization agent"""
    chart: go.Figure
    data: pd.DataFrame
    insights: List[str]
    narrative: str
    chart_type: ChartType


class FinancialVisualizationAgent:
    """Agent that extracts data and creates financial visualizations"""

    def __init__(self, rag_system, openai_client: OpenAI):
        self.rag_system = rag_system
        self.client = openai_client
        self.model = "gpt-4o-mini"

    def extract_financial_data(self, query: str) -> Dict[str, Any]:
        """Extract structured financial data from documents"""

        # Step 1: Identify what data we need
        data_identification_prompt = f"""You are a financial data analyst. Analyze this query and identify what financial data needs to be extracted.

Query: {query}

Identify:
1. What metrics to extract (e.g., revenue, net income, operating expenses)
2. What time periods to look for (e.g., 2023, 2022, 2021, quarterly data)
3. What comparisons are needed

Return JSON:
{{
    "metrics": ["metric1", "metric2"],
    "search_queries": ["specific query 1", "specific query 2"],
    "expected_periods": ["2023", "2022", "2021"],
    "chart_type": "line" or "bar" or "pie" or "waterfall"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial data analyst. Output only valid JSON."},
                    {"role": "user", "content": data_identification_prompt}
                ],
                temperature=0.2
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            data_spec = json.loads(result_text)
        except:
            # Fallback
            data_spec = {
                "metrics": ["revenue"],
                "search_queries": [query],
                "expected_periods": ["2023", "2022", "2021"],
                "chart_type": "line"
            }

        # Step 2: Retrieve relevant financial data sections
        all_context = []
        for search_query in data_spec.get("search_queries", [query]):
            retrieved = self.rag_system.hybrid_retrieve(search_query, top_k=8)
            for chunk in retrieved:
                all_context.append(chunk['text'])

        combined_context = "\n\n".join(all_context[:10])  # Limit context

        # Step 3: Extract structured data using LLM
        extraction_prompt = f"""Extract financial data from these documents. Find numerical values for the specified metrics across different time periods.

Metrics to extract: {', '.join(data_spec.get('metrics', []))}
Requested periods (user asked for these, but extract whatever is available): {', '.join(data_spec.get('expected_periods', []))}

Context from 10-K filing:
{combined_context}

Extract data and return as JSON:
{{
    "data_points": [
        {{"metric": "Revenue", "value": 394328, "period": "2023", "unit": "millions"}},
        {{"metric": "Revenue", "value": 365817, "period": "2022", "unit": "millions"}},
        ...
    ],
    "source_confidence": "high" or "medium" or "low",
    "notes": "any important notes about the data, including which periods were requested but not found"
}}

CRITICAL INSTRUCTIONS:
- Be precise with numbers. If a value is stated as "394.3 billion", convert to millions (394300).
- Extract ALL available data points, even if some requested periods are missing. Partial data is acceptable.
- If a value is not found for a specific period, DO NOT include it in data_points (don't use null).
- Include ALL periods where data IS available, even if they're not in the expected_periods list.
- In the "notes" field, clearly state which requested periods were not found (e.g., "2021 and 2022 data not found in document").
- Convert all values to the same unit (preferably millions) for consistency.
- Prioritize extracting available data over perfect period matching - it's better to show what exists than nothing."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial data extraction expert. Output only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            extracted_data = json.loads(result_text)
            extracted_data['chart_type'] = data_spec.get('chart_type', 'line')
            return extracted_data
        except Exception as e:
            return {
                "data_points": [],
                "source_confidence": "low",
                "notes": f"Error extracting data: {str(e)}",
                "chart_type": "line"
            }

    def create_chart(self, data_points: List[Dict], chart_type: str, query: str, missing_periods_note: str = "") -> go.Figure:
        """Create interactive Plotly chart from data points"""

        if not data_points or len(data_points) == 0:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Convert to DataFrame
        df = pd.DataFrame(data_points)

        # Filter out null values
        df = df[df['value'].notna()]

        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid data points found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig

        # Determine chart type
        chart_type_enum = ChartType(chart_type.lower()) if chart_type else ChartType.LINE

        if chart_type_enum == ChartType.LINE:
            fig = self._create_line_chart(df, query)
        elif chart_type_enum == ChartType.BAR:
            fig = self._create_bar_chart(df, query)
        elif chart_type_enum == ChartType.WATERFALL:
            fig = self._create_waterfall_chart(df, query)
        elif chart_type_enum == ChartType.PIE:
            fig = self._create_pie_chart(df, query)
        elif chart_type_enum == ChartType.AREA:
            fig = self._create_area_chart(df, query)
        else:
            fig = self._create_line_chart(df, query)

        # Common layout improvements - add note to existing title if needed
        if missing_periods_note:
            # Get current title if it exists, otherwise use query
            current_title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else query
            # Append note as subtitle
            title_text = f"{current_title}<br><sub style='font-size: 11px; color: #888;'>{missing_periods_note}</sub>"
            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    xanchor='center'
                ),
                margin=dict(t=100)  # Increase top margin for subtitle
            )
        
        # Apply common styling
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=500,
            margin=dict(l=60, r=60, t=100 if missing_periods_note else 80, b=60)
        )

        return fig

    def _create_line_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create line chart for trends"""
        fig = go.Figure()

        # Group by metric
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('period')

            fig.add_trace(go.Scatter(
                x=metric_data['period'],
                y=metric_data['value'],
                mode='lines+markers',
                name=metric,
                line=dict(width=3),
                marker=dict(size=10),
                hovertemplate=f'<b>{metric}</b><br>' +
                             'Period: %{x}<br>' +
                             'Value: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))

        # Get unit from first data point
        unit = df['unit'].iloc[0] if 'unit' in df.columns else 'USD'

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title="Period",
            yaxis_title=f"Amount ({unit})",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def _create_bar_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create bar chart for comparisons"""
        fig = go.Figure()

        # Group by metric
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('period')

            fig.add_trace(go.Bar(
                x=metric_data['period'],
                y=metric_data['value'],
                name=metric,
                hovertemplate=f'<b>{metric}</b><br>' +
                             'Period: %{x}<br>' +
                             'Value: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))

        unit = df['unit'].iloc[0] if 'unit' in df.columns else 'USD'

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title="Period",
            yaxis_title=f"Amount ({unit})",
            barmode='group',
            showlegend=True
        )

        return fig

    def _create_area_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create area chart for cumulative view"""
        fig = go.Figure()

        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric].sort_values('period')

            fig.add_trace(go.Scatter(
                x=metric_data['period'],
                y=metric_data['value'],
                mode='lines',
                name=metric,
                fill='tonexty',
                line=dict(width=2),
                hovertemplate=f'<b>{metric}</b><br>' +
                             'Period: %{x}<br>' +
                             'Value: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))

        unit = df['unit'].iloc[0] if 'unit' in df.columns else 'USD'

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title="Period",
            yaxis_title=f"Amount ({unit})",
            showlegend=True
        )

        return fig

    def _create_waterfall_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create waterfall chart for showing changes"""
        # Sort by period
        df_sorted = df.sort_values('period')

        # Calculate differences
        values = df_sorted['value'].tolist()
        periods = df_sorted['period'].tolist()

        measures = ["absolute"] + ["relative"] * (len(values) - 1)

        fig = go.Figure(go.Waterfall(
            x=periods,
            y=values,
            measure=measures,
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        unit = df['unit'].iloc[0] if 'unit' in df.columns else 'USD'

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title="Period",
            yaxis_title=f"Amount ({unit})",
            showlegend=False
        )

        return fig

    def _create_pie_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create pie chart for composition"""
        # Use most recent period or aggregate
        if 'period' in df.columns:
            latest_period = df['period'].max()
            df_filtered = df[df['period'] == latest_period]
        else:
            df_filtered = df

        fig = go.Figure(data=[go.Pie(
            labels=df_filtered['metric'],
            values=df_filtered['value'],
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )])

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            )
        )

        return fig

    def generate_insights(self, df: pd.DataFrame, query: str) -> List[str]:
        """Generate insights from the data"""

        insights_prompt = f"""Analyze this financial data and provide 3-5 key insights.

Query: {query}

Data:
{df.to_string()}

Provide insights about:
- Trends (increasing, decreasing, stable)
- Growth rates or percentage changes
- Notable patterns or anomalies
- Comparative analysis

Return as JSON array:
["insight 1", "insight 2", "insight 3"]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Output only valid JSON array."},
                    {"role": "user", "content": insights_prompt}
                ],
                temperature=0.3
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            return json.loads(result_text)
        except:
            # Calculate basic insights
            insights = []

            if 'period' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('period')
                first_val = df_sorted['value'].iloc[0]
                last_val = df_sorted['value'].iloc[-1]
                change_pct = ((last_val - first_val) / first_val) * 100

                insights.append(f"Overall change: {change_pct:+.1f}% from {df_sorted['period'].iloc[0]} to {df_sorted['period'].iloc[-1]}")

            return insights

    def create_narrative(self, df: pd.DataFrame, insights: List[str], query: str, notes: str = "") -> str:
        """Create narrative explanation of the visualization"""

        data_info = f"Available periods: {', '.join(sorted(df['period'].unique().astype(str))) if not df.empty and 'period' in df.columns else 'None'}"
        
        narrative_prompt = f"""Create a brief narrative explanation of this financial visualization.

Query: {query}

Data Summary:
{df.describe().to_string() if not df.empty else 'No data available'}

{data_info}

Key Insights:
{chr(10).join(f'- {i}' for i in insights) if insights else 'None'}

Additional Notes: {notes if notes else 'None'}

Write a 2-3 paragraph narrative that:
1. Describes what the chart shows and which time periods are included
2. Highlights the most important findings and trends
3. If some requested periods are missing, acknowledge this but focus on the available data
4. Provides context for the trends shown

Keep it professional and concise (150-200 words)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial writer creating clear explanations."},
                    {"role": "user", "content": narrative_prompt}
                ],
                temperature=0.4
            )

            return response.choices[0].message.content.strip()
        except:
            return f"The visualization shows financial data related to: {query}. " + \
                   " ".join(insights[:2])

    def run(self, query: str) -> VisualizationResult:
        """Main execution: extract data, create chart, generate insights"""

        # Step 1: Extract financial data
        extracted = self.extract_financial_data(query)

        data_points = extracted.get('data_points', [])
        chart_type = extracted.get('chart_type', 'line')

        # Step 2: Create DataFrame
        if data_points:
            df = pd.DataFrame(data_points)
            df = df[df['value'].notna()]  # Remove null values
        else:
            df = pd.DataFrame()

        # Step 3: Extract missing periods info for chart note
        notes = extracted.get('notes', '')
        missing_note = ""
        if notes and ("not found" in notes.lower() or "missing" in notes.lower() or "not available" in notes.lower()):
            # Extract which periods were requested but not found
            missing_note = f"Note: {notes}" if len(notes) < 100 else f"Note: Some requested periods may not be available in the document."

        # Step 4: Create chart
        chart = self.create_chart(data_points, chart_type, query, missing_note)

        # Step 5: Generate insights
        if not df.empty:
            insights = self.generate_insights(df, query)
        else:
            insights = ["No data available for visualization"]

        # Step 6: Create narrative
        if not df.empty:
            narrative = self.create_narrative(df, insights, query, notes)
        else:
            missing_info = f" Requested periods may not be available in the document." if notes else ""
            narrative = f"Unable to extract sufficient financial data from the documents to create a meaningful visualization.{missing_info} Try refining your query or ensure the 10-K filing contains the requested information."

        return VisualizationResult(
            chart=chart,
            data=df,
            insights=insights,
            narrative=narrative,
            chart_type=ChartType(chart_type)
        )
