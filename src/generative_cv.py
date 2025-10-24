#!/usr/bin/env python3
"""
A-Qlegal Generative Computer Vision Module
Advanced visual content generation for legal documents and presentations

Features:
- Legal document template generation with visual layouts
- Legal diagram and flowchart generation
- Legal infographic and visual content generation
- Contract visualization and layout generation
- Legal process flow diagrams
- Court document templates with visual elements

Author: A-Qlegal Team
Version: 1.0.0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
import io
import base64
from datetime import datetime
import random
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Try to import optional libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some advanced visualizations will be limited.")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


@dataclass
class LegalDocumentLayout:
    """Data class for legal document layout specifications"""
    title: str
    sections: List[Dict[str, Any]]
    signature_blocks: List[Dict[str, Any]]
    watermark: Optional[str] = None
    header_footer: Dict[str, str] = None
    styling: Dict[str, Any] = None


@dataclass
class LegalDiagramConfig:
    """Configuration for legal diagrams"""
    diagram_type: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    styling: Dict[str, Any]
    title: str
    description: str


class GenerativeComputerVision:
    """Advanced generative computer vision for legal applications"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f4e79',
            'secondary': '#2e7d32',
            'accent': '#d32f2f',
            'neutral': '#424242',
            'light': '#f5f5f5',
            'dark': '#212121',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'info': '#2196f3'
        }
        
        self.legal_colors = {
            'court': '#8e24aa',
            'law_firm': '#1976d2',
            'government': '#388e3c',
            'corporate': '#f57c00',
            'criminal': '#d32f2f',
            'civil': '#00796b',
            'family': '#7b1fa2',
            'property': '#5d4037',
            'contract': '#303f9f',
            'constitutional': '#689f38'
        }
        
        # Initialize matplotlib style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except Exception:
            pass  # Use default palette if seaborn fails
    
    def generate_legal_document_template(self, 
                                       document_type: str, 
                                       content: Dict[str, Any],
                                       layout_style: str = "professional") -> Image.Image:
        """
        Generate a visual legal document template with proper layout and styling
        
        Args:
            document_type: Type of legal document (contract, affidavit, etc.)
            content: Document content with sections and text
            layout_style: Visual style (professional, modern, traditional)
        
        Returns:
            PIL Image of the generated document template
        """
        # Document dimensions (A4 size)
        width, height = 2480, 3508  # 300 DPI A4
        
        # Create base image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts (fallback to default if not available)
        title_font = self._load_font(48)
        header_font = self._load_font(32)
        body_font = self._load_font(24)
        small_font = self._load_font(18)
        
        # Define layout parameters
        margin = 200
        line_height = 40
        section_spacing = 60
        
        y_position = margin
        
        # Draw header with document type
        header_bg = self._get_document_color(document_type)
        draw.rectangle([0, 0, width, 150], fill=header_bg)
        
        # Document title
        title_text = content.get('title', f'{document_type.upper()} DOCUMENT')
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 50), title_text, fill='white', font=title_font)
        
        # Document metadata
        metadata = content.get('metadata', {})
        if metadata:
            date_text = f"Date: {metadata.get('date', datetime.now().strftime('%B %d, %Y'))}"
            draw.text((margin, 120), date_text, fill='white', font=small_font)
            
            if 'case_number' in metadata:
                case_text = f"Case Number: {metadata['case_number']}"
                case_bbox = draw.textbbox((0, 0), case_text, font=small_font)
                case_width = case_bbox[2] - case_bbox[0]
                draw.text((width - margin - case_width, 120), case_text, fill='white', font=small_font)
        
        y_position = 200
        
        # Draw document sections
        sections = content.get('sections', [])
        for section in sections:
            y_position = self._draw_section(draw, section, y_position, width, margin, 
                                          header_font, body_font, line_height, section_spacing)
        
        # Draw signature blocks
        signature_blocks = content.get('signature_blocks', [])
        if signature_blocks:
            y_position += section_spacing
            y_position = self._draw_signature_blocks(draw, signature_blocks, y_position, 
                                                   width, margin, header_font, body_font)
        
        # Draw footer
        footer_y = height - 100
        draw.rectangle([0, footer_y, width, height], fill=self.colors['light'])
        footer_text = f"Generated by A-Qlegal AI System - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        footer_bbox = draw.textbbox((0, 0), footer_text, font=small_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        footer_x = (width - footer_width) // 2
        draw.text((footer_x, footer_y + 30), footer_text, fill=self.colors['neutral'], font=small_font)
        
        # Add watermark if specified
        watermark = content.get('watermark', '')
        if watermark:
            self._add_watermark(img, watermark)
        
        return img
    
    def generate_legal_diagram(self, 
                             diagram_config: LegalDiagramConfig,
                             style: str = "professional") -> Image.Image:
        """
        Generate legal process diagrams, flowcharts, and organizational charts
        
        Args:
            diagram_config: Configuration for the diagram
            style: Visual style (professional, modern, colorful)
        
        Returns:
            PIL Image of the generated diagram
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Set title
        ax.text(5, 7.5, diagram_config.title, 
               fontsize=24, fontweight='bold', ha='center',
               color=self.colors['primary'])
        
        if diagram_config.description:
            ax.text(5, 7, diagram_config.description, 
                   fontsize=14, ha='center', style='italic',
                   color=self.colors['neutral'])
        
        # Draw nodes
        for node in diagram_config.nodes:
            self._draw_diagram_node(ax, node, diagram_config.styling)
        
        # Draw edges
        for edge in diagram_config.edges:
            self._draw_diagram_edge(ax, edge, diagram_config.styling)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def generate_legal_infographic(self, 
                                 data: Dict[str, Any],
                                 infographic_type: str = "statistics") -> Image.Image:
        """
        Generate legal infographics with statistics, timelines, and visual data
        
        Args:
            data: Data to visualize
            infographic_type: Type of infographic (statistics, timeline, comparison)
        
        Returns:
            PIL Image of the generated infographic
        """
        if not PLOTLY_AVAILABLE:
            return self._generate_simple_infographic(data, infographic_type)
        
        # Create subplots based on infographic type
        if infographic_type == "statistics":
            return self._generate_statistics_infographic(data)
        elif infographic_type == "timeline":
            return self._generate_timeline_infographic(data)
        elif infographic_type == "comparison":
            return self._generate_comparison_infographic(data)
        else:
            return self._generate_custom_infographic(data, infographic_type)
    
    def generate_contract_visualization(self, 
                                      contract_data: Dict[str, Any],
                                      visualization_type: str = "structure") -> Image.Image:
        """
        Generate visual representations of contracts and legal agreements
        
        Args:
            contract_data: Contract information and clauses
            visualization_type: Type of visualization (structure, timeline, risk)
        
        Returns:
            PIL Image of the contract visualization
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        title = contract_data.get('title', 'Contract Visualization')
        ax.text(5, 7.5, title, fontsize=20, fontweight='bold', ha='center',
               color=self.colors['primary'])
        
        if visualization_type == "structure":
            return self._generate_contract_structure(contract_data, fig, ax)
        elif visualization_type == "timeline":
            return self._generate_contract_timeline(contract_data, fig, ax)
        elif visualization_type == "risk":
            return self._generate_contract_risk_analysis(contract_data, fig, ax)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def generate_legal_flowchart(self, 
                               process_steps: List[Dict[str, Any]],
                               flowchart_type: str = "legal_process") -> Image.Image:
        """
        Generate legal process flowcharts and decision trees
        
        Args:
            process_steps: List of process steps with conditions and outcomes
            flowchart_type: Type of flowchart (legal_process, decision_tree, court_process)
        
        Returns:
            PIL Image of the generated flowchart
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Calculate layout
        layout = self._calculate_flowchart_layout(process_steps)
        
        # Draw flowchart elements
        for i, step in enumerate(process_steps):
            x, y = layout[i]
            self._draw_flowchart_step(ax, step, x, y)
        
        # Draw connections
        self._draw_flowchart_connections(ax, process_steps, layout)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def generate_court_document_template(self, 
                                       document_type: str,
                                       case_info: Dict[str, Any]) -> Image.Image:
        """
        Generate official court document templates with proper formatting
        
        Args:
            document_type: Type of court document
            case_info: Case information and details
        
        Returns:
            PIL Image of the court document template
        """
        width, height = 2480, 3508
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        title_font = self._load_font(42)
        header_font = self._load_font(28)
        body_font = self._load_font(22)
        small_font = self._load_font(16)
        
        margin = 200
        y_position = margin
        
        # Court header
        court_name = case_info.get('court_name', 'UNITED STATES DISTRICT COURT')
        draw.rectangle([0, 0, width, 120], fill=self.colors['primary'])
        draw.text((margin, 40), court_name, fill='white', font=header_font)
        
        # Case information
        y_position = 180
        case_number = case_info.get('case_number', 'Case No. 12345')
        draw.text((margin, y_position), case_number, fill=self.colors['dark'], font=body_font)
        
        y_position += 60
        parties = case_info.get('parties', 'Plaintiff vs. Defendant')
        draw.text((margin, y_position), parties, fill=self.colors['dark'], font=body_font)
        
        # Document title
        y_position += 100
        doc_title = f"{document_type.upper()}"
        title_bbox = draw.textbbox((0, 0), doc_title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, y_position), doc_title, fill=self.colors['primary'], font=title_font)
        
        # Document content area
        y_position += 120
        content_area = case_info.get('content', 'Document content will be placed here.')
        
        # Draw content lines
        lines = self._wrap_text(content_area, width - 2 * margin, body_font)
        for line in lines:
            if y_position < height - 200:
                draw.text((margin, y_position), line, fill=self.colors['dark'], font=body_font)
                y_position += 35
        
        # Signature lines
        y_position = height - 300
        draw.text((margin, y_position), "_________________________", fill=self.colors['dark'], font=body_font)
        draw.text((margin, y_position + 40), "Judge Signature", fill=self.colors['neutral'], font=small_font)
        
        draw.text((width - margin - 200, y_position), "_________________________", fill=self.colors['dark'], font=body_font)
        draw.text((width - margin - 200, y_position + 40), "Date", fill=self.colors['neutral'], font=small_font)
        
        return img
    
    def generate_legal_presentation_slide(self, 
                                        slide_data: Dict[str, Any],
                                        slide_type: str = "content") -> Image.Image:
        """
        Generate legal presentation slides with visual elements
        
        Args:
            slide_data: Slide content and data
            slide_type: Type of slide (title, content, chart, comparison)
        
        Returns:
            PIL Image of the presentation slide
        """
        width, height = 1920, 1080  # HD resolution
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        title_font = self._load_font(48)
        header_font = self._load_font(32)
        body_font = self._load_font(24)
        small_font = self._load_font(18)
        
        # Background gradient
        self._draw_gradient_background(draw, width, height)
        
        margin = 100
        y_position = margin
        
        # Slide title
        title = slide_data.get('title', 'Legal Presentation')
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, y_position), title, fill=self.colors['primary'], font=title_font)
        
        y_position += 120
        
        # Slide content based on type
        if slide_type == "content":
            self._draw_content_slide(draw, slide_data, y_position, width, margin, 
                                   header_font, body_font)
        elif slide_type == "chart":
            self._draw_chart_slide(draw, slide_data, y_position, width, margin)
        elif slide_type == "comparison":
            self._draw_comparison_slide(draw, slide_data, y_position, width, margin, 
                                      header_font, body_font)
        
        return img
    
    # Helper methods
    def _load_font(self, size: int):
        """Load font with fallback options"""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            # Try alternative font paths
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Arial.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    return ImageFont.truetype(font_path, size)
                except (OSError, IOError):
                    continue
            
            return ImageFont.load_default()
    
    def _get_document_color(self, document_type: str) -> str:
        """Get appropriate color for document type"""
        color_map = {
            'contract': self.legal_colors['contract'],
            'affidavit': self.legal_colors['court'],
            'motion': self.legal_colors['criminal'],
            'brief': self.legal_colors['civil'],
            'petition': self.legal_colors['family'],
            'order': self.legal_colors['government'],
            'default': self.colors['primary']
        }
        return color_map.get(document_type.lower(), color_map['default'])
    
    def _draw_section(self, draw, section: Dict[str, Any], y_pos: int, width: int, 
                     margin: int, header_font, body_font, line_height: int, 
                     section_spacing: int) -> int:
        """Draw a document section"""
        section_title = section.get('title', '')
        section_content = section.get('content', '')
        
        if section_title:
            draw.text((margin, y_pos), section_title, fill=self.colors['primary'], font=header_font)
            y_pos += line_height + 20
        
        # Wrap and draw content
        lines = self._wrap_text(section_content, width - 2 * margin, body_font)
        for line in lines:
            draw.text((margin + 40, y_pos), line, fill=self.colors['dark'], font=body_font)
            y_pos += line_height
        
        return y_pos + section_spacing
    
    def _draw_signature_blocks(self, draw, signature_blocks: List[Dict[str, Any]], 
                             y_pos: int, width: int, margin: int, header_font, body_font) -> int:
        """Draw signature blocks"""
        for block in signature_blocks:
            block_title = block.get('title', 'Signature')
            draw.text((margin, y_pos), f"{block_title}:", fill=self.colors['primary'], font=header_font)
            y_pos += 60
            
            # Signature line
            draw.line([(margin, y_pos), (margin + 300, y_pos)], fill=self.colors['dark'], width=2)
            y_pos += 80
            
            # Name line
            name = block.get('name', '')
            draw.text((margin, y_pos), name, fill=self.colors['dark'], font=body_font)
            y_pos += 40
            
            # Date line
            date_text = f"Date: {block.get('date', datetime.now().strftime('%B %d, %Y'))}"
            draw.text((margin, y_pos), date_text, fill=self.colors['neutral'], font=body_font)
            y_pos += 80
        
        return y_pos
    
    def _add_watermark(self, img: Image.Image, watermark_text: str):
        """Add watermark to image"""
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        watermark_font = self._load_font(60)
        
        # Semi-transparent watermark
        watermark_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
        watermark_draw = ImageDraw.Draw(watermark_img)
        
        # Calculate position for diagonal watermark
        bbox = watermark_draw.textbbox((0, 0), watermark_text, font=watermark_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw watermark at 45 degree angle
        for i in range(0, width + text_width, text_width + 100):
            for j in range(0, height + text_height, text_height + 100):
                watermark_draw.text((i, j), watermark_text, 
                                  fill=(128, 128, 128, 50), font=watermark_font)
        
        # Composite watermark onto main image
        img.paste(watermark_img, (0, 0), watermark_img)
    
    def _wrap_text(self, text: str, max_width: int, font) -> List[str]:
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            
            # Create a temporary image to measure text width
            temp_img = Image.new('RGB', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width > max_width:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_diagram_node(self, ax, node: Dict[str, Any], styling: Dict[str, Any]):
        """Draw a diagram node"""
        x, y = node['position']
        label = node['label']
        node_type = node.get('type', 'rectangle')
        
        if node_type == 'rectangle':
            rect = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['light'],
                                edgecolor=self.colors['primary'],
                                linewidth=2)
            ax.add_patch(rect)
        elif node_type == 'circle':
            circle = Circle((x, y), 0.3, facecolor=self.colors['light'],
                          edgecolor=self.colors['primary'], linewidth=2)
            ax.add_patch(circle)
        elif node_type == 'diamond':
            diamond = patches.RegularPolygon((x, y), 4, radius=0.4,
                                           orientation=np.pi/4,
                                           facecolor=self.colors['light'],
                                           edgecolor=self.colors['primary'],
                                           linewidth=2)
            ax.add_patch(diamond)
        
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_diagram_edge(self, ax, edge: Dict[str, Any], styling: Dict[str, Any]):
        """Draw a diagram edge/connection"""
        start = edge['start']
        end = edge['end']
        label = edge.get('label', '')
        
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['primary']))
        
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def _generate_statistics_infographic(self, data: Dict[str, Any]) -> Image.Image:
        """Generate statistics infographic using Plotly or matplotlib fallback"""
        try:
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Case Statistics', 'Success Rate', 'Timeline', 'Categories'),
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"type": "scatter"}, {"type": "bar"}]]
                )
                
                # Bar chart for case statistics
                if 'case_stats' in data:
                    fig.add_trace(
                        go.Bar(x=list(data['case_stats'].keys()), 
                              y=list(data['case_stats'].values()),
                              name="Cases", marker_color=self.colors['primary']),
                        row=1, col=1
                    )
                
                # Pie chart for success rate
                if 'success_rate' in data:
                    fig.add_trace(
                        go.Pie(labels=['Successful', 'Unsuccessful'], 
                              values=[data['success_rate'], 100-data['success_rate']],
                              name="Success Rate"),
                        row=1, col=2
                    )
                
                # Scatter plot for timeline
                if 'timeline' in data:
                    timeline_data = data['timeline']
                    fig.add_trace(
                        go.Scatter(x=timeline_data['dates'], y=timeline_data['values'],
                                  mode='lines+markers', name="Timeline"),
                        row=2, col=1
                    )
                
                # Bar chart for categories
                if 'categories' in data:
                    fig.add_trace(
                        go.Bar(x=list(data['categories'].keys()), 
                              y=list(data['categories'].values()),
                              name="Categories", marker_color=self.colors['secondary']),
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, showlegend=False, title_text="Legal Statistics Dashboard")
                
                # Convert to PIL Image
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                img = Image.open(io.BytesIO(img_bytes))
                return img
            else:
                raise ImportError("Plotly not available")
        except Exception as e:
            print(f"Plotly infographic failed, using matplotlib fallback: {e}")
            return self._generate_simple_infographic(data, "statistics")
    
    def _generate_timeline_infographic(self, data: Dict[str, Any]) -> Image.Image:
        """Generate timeline infographic"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        events = data.get('events', [])
        if not events:
            events = [
                {'date': '2023-01-01', 'title': 'Case Filed', 'description': 'Initial filing'},
                {'date': '2023-03-15', 'title': 'Discovery', 'description': 'Evidence gathering'},
                {'date': '2023-06-01', 'title': 'Trial', 'description': 'Court proceedings'},
                {'date': '2023-08-15', 'title': 'Verdict', 'description': 'Final decision'}
            ]
        
        # Draw timeline
        y_positions = np.linspace(1, 7, len(events))
        
        for i, event in enumerate(events):
            # Draw timeline node
            circle = Circle((i, y_positions[i]), 0.2, 
                          facecolor=self.colors['primary'], 
                          edgecolor=self.colors['dark'])
            ax.add_patch(circle)
            
            # Draw event text
            ax.text(i + 0.3, y_positions[i], event['title'], 
                   fontsize=12, fontweight='bold', va='center')
            ax.text(i + 0.3, y_positions[i] - 0.3, event['description'], 
                   fontsize=10, va='center', style='italic')
            
            # Draw connecting line
            if i < len(events) - 1:
                ax.plot([i + 0.2, i + 0.8], [y_positions[i], y_positions[i+1]], 
                       'k-', alpha=0.5, linewidth=2)
        
        ax.set_xlim(-0.5, len(events) - 0.5)
        ax.set_ylim(0, 8)
        ax.set_title('Legal Process Timeline', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _generate_comparison_infographic(self, data: Dict[str, Any]) -> Image.Image:
        """Generate comparison infographic"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Handle both dict and list formats for categories
        categories_data = data.get('categories', ['Category A', 'Category B', 'Category C'])
        if isinstance(categories_data, dict):
            categories = list(categories_data.keys())
            values_a = list(categories_data.values())
            values_b = [v * 0.8 for v in values_a]  # Create comparison data
        else:
            categories = categories_data
            values_a = data.get('values_a', [30, 45, 25])
            values_b = data.get('values_b', [25, 50, 35])
        
        # Ensure all arrays have the same length
        min_length = min(len(categories), len(values_a), len(values_b))
        categories = categories[:min_length]
        values_a = values_a[:min_length]
        values_b = values_b[:min_length]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values_a, width, label='Group A', color=self.colors['primary'])
        bars2 = ax.bar(x + width/2, values_b, width, label='Group B', color=self.colors['secondary'])
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_title('Legal Comparison Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _generate_simple_infographic(self, data: Dict[str, Any], infographic_type: str) -> Image.Image:
        """Generate simple infographic without Plotly"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if infographic_type == "statistics":
            categories = list(data.get('categories', {}).keys())
            values = list(data.get('categories', {}).values())
            
            if not categories:
                categories = ['Civil', 'Criminal', 'Family', 'Corporate']
                values = [30, 25, 20, 25]
            
            bars = ax.bar(categories, values, color=[self.colors['primary'], self.colors['secondary'], 
                                                   self.colors['accent'], self.colors['success']])
            ax.set_title('Legal Case Categories', fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Cases')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _generate_custom_infographic(self, data: Dict[str, Any], infographic_type: str) -> Image.Image:
        """Generate custom infographic based on type"""
        return self._generate_simple_infographic(data, infographic_type)
    
    def _generate_contract_structure(self, contract_data: Dict[str, Any], fig, ax) -> Image.Image:
        """Generate contract structure visualization"""
        clauses = contract_data.get('clauses', [])
        
        # Draw contract structure
        y_pos = 6
        for i, clause in enumerate(clauses):
            # Draw clause box
            rect = FancyBboxPatch((1, y_pos - 0.4), 8, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['light'],
                                edgecolor=self.colors['primary'],
                                linewidth=2)
            ax.add_patch(rect)
            
            # Draw clause text
            ax.text(5, y_pos, clause.get('title', f'Clause {i+1}'), 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Draw connection to next clause
            if i < len(clauses) - 1:
                ax.arrow(5, y_pos - 0.4, 0, -0.6, head_width=0.2, head_length=0.1, 
                        fc=self.colors['primary'], ec=self.colors['primary'])
            
            y_pos -= 1.5
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _generate_contract_timeline(self, contract_data: Dict[str, Any], fig, ax) -> Image.Image:
        """Generate contract timeline visualization"""
        milestones = contract_data.get('milestones', [])
        
        # Draw timeline
        ax.plot([1, 9], [4, 4], 'k-', linewidth=3, alpha=0.7)
        
        for i, milestone in enumerate(milestones):
            x_pos = 1 + (i * 8 / (len(milestones) - 1)) if len(milestones) > 1 else 5
            
            # Draw milestone marker
            circle = Circle((x_pos, 4), 0.3, facecolor=self.colors['primary'], 
                          edgecolor=self.colors['dark'], linewidth=2)
            ax.add_patch(circle)
            
            # Draw milestone text
            ax.text(x_pos, 4.8, milestone.get('title', f'Milestone {i+1}'), 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(x_pos, 3.2, milestone.get('date', ''), 
                   ha='center', va='center', fontsize=8, style='italic')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _generate_contract_risk_analysis(self, contract_data: Dict[str, Any], fig, ax) -> Image.Image:
        """Generate contract risk analysis visualization"""
        risks = contract_data.get('risks', [])
        
        # Create risk matrix
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        probabilities = ['Low', 'Medium', 'High']
        
        # Draw risk matrix
        for i, risk in enumerate(risks):
            risk_level = risk.get('level', 'Medium')
            probability = risk.get('probability', 'Medium')
            
            # Map risk levels to coordinates
            risk_y = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}[risk_level]
            prob_x = {'Low': 1, 'Medium': 2, 'High': 3}[probability]
            
            # Draw risk point
            circle = Circle((prob_x, risk_y), 0.2, 
                          facecolor=self.colors['accent'], 
                          edgecolor=self.colors['dark'])
            ax.add_patch(circle)
            
            # Draw risk label
            ax.text(prob_x + 0.3, risk_y, risk.get('title', f'Risk {i+1}'), 
                   va='center', fontsize=9)
        
        # Add axis labels
        ax.set_xlabel('Probability')
        ax.set_ylabel('Risk Level')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(probabilities)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(risk_levels)
        ax.set_title('Contract Risk Analysis Matrix')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def _calculate_flowchart_layout(self, process_steps: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Calculate layout positions for flowchart elements"""
        positions = []
        num_steps = len(process_steps)
        
        if num_steps <= 4:
            # Single row layout
            for i in range(num_steps):
                x = 2 + (i * 2.5)
                y = 5
                positions.append((x, y))
        else:
            # Multi-row layout
            cols = 3
            for i in range(num_steps):
                row = i // cols
                col = i % cols
                x = 2 + (col * 3)
                y = 7 - (row * 2)
                positions.append((x, y))
        
        return positions
    
    def _draw_flowchart_step(self, ax, step: Dict[str, Any], x: float, y: float):
        """Draw a flowchart step"""
        step_type = step.get('type', 'process')
        label = step.get('label', 'Step')
        
        if step_type == 'process':
            rect = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['light'],
                                edgecolor=self.colors['primary'],
                                linewidth=2)
            ax.add_patch(rect)
        elif step_type == 'decision':
            diamond = patches.RegularPolygon((x, y), 4, radius=0.6,
                                           orientation=np.pi/4,
                                           facecolor=self.colors['light'],
                                           edgecolor=self.colors['accent'],
                                           linewidth=2)
            ax.add_patch(diamond)
        elif step_type == 'start_end':
            circle = Circle((x, y), 0.4, facecolor=self.colors['success'], 
                          edgecolor=self.colors['dark'], linewidth=2)
            ax.add_patch(circle)
        
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_flowchart_connections(self, ax, process_steps: List[Dict[str, Any]], layout: List[Tuple[float, float]]):
        """Draw connections between flowchart steps"""
        for i in range(len(process_steps) - 1):
            start = layout[i]
            end = layout[i + 1]
            
            # Calculate arrow direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Draw arrow
            ax.arrow(start[0], start[1] - 0.3, dx, dy - 0.4, 
                    head_width=0.15, head_length=0.1, 
                    fc=self.colors['primary'], ec=self.colors['primary'])
    
    def _draw_gradient_background(self, draw, width: int, height: int):
        """Draw gradient background"""
        # Convert hex colors to RGB
        light_rgb = tuple(int(self.colors['light'][i:i+2], 16) for i in range(1, 7, 2))
        primary_rgb = tuple(int(self.colors['primary'][i:i+2], 16) for i in range(1, 7, 2))
        
        for y in range(height):
            ratio = y / height
            color = tuple(
                int(light_rgb[i] + (primary_rgb[i] - light_rgb[i]) * ratio)
                for i in range(3)
            )
            draw.line([(0, y), (width, y)], fill=color)
    
    def _draw_content_slide(self, draw, slide_data: Dict[str, Any], y_pos: int, 
                          width: int, margin: int, header_font, body_font):
        """Draw content slide"""
        content_items = slide_data.get('content', [])
        
        for item in content_items:
            if item['type'] == 'heading':
                draw.text((margin, y_pos), item['text'], fill=self.colors['primary'], font=header_font)
                y_pos += 60
            elif item['type'] == 'bullet':
                draw.text((margin + 40, y_pos), f"• {item['text']}", fill=self.colors['dark'], font=body_font)
                y_pos += 40
            elif item['type'] == 'paragraph':
                lines = self._wrap_text(item['text'], width - 2 * margin, body_font)
                for line in lines:
                    draw.text((margin, y_pos), line, fill=self.colors['dark'], font=body_font)
                    y_pos += 35
                y_pos += 20
    
    def _draw_chart_slide(self, draw, slide_data: Dict[str, Any], y_pos: int, width: int, margin: int):
        """Draw chart slide"""
        chart_data = slide_data.get('chart_data', {})
        
        # Load fonts for chart labels
        body_font = self._load_font(20)
        small_font = self._load_font(16)
        
        # Simple bar chart representation
        if 'categories' in chart_data and 'values' in chart_data:
            categories = chart_data['categories']
            values = chart_data['values']
            
            chart_width = width - 2 * margin
            chart_height = 300
            bar_width = chart_width // len(categories)
            
            max_value = max(values) if values else 1
            
            for i, (category, value) in enumerate(zip(categories, values)):
                bar_height = int((value / max_value) * chart_height)
                bar_x = margin + (i * bar_width)
                bar_y = y_pos + chart_height - bar_height
                
                # Draw bar
                draw.rectangle([bar_x, bar_y, bar_x + bar_width - 10, y_pos + chart_height], 
                             fill=self.colors['primary'])
                
                # Draw value label
                draw.text((bar_x + 5, bar_y - 25), str(value), fill=self.colors['dark'], font=body_font)
                
                # Draw category label
                draw.text((bar_x + 5, y_pos + chart_height + 10), category, 
                         fill=self.colors['dark'], font=small_font)
    
    def _draw_comparison_slide(self, draw, slide_data: Dict[str, Any], y_pos: int, 
                             width: int, margin: int, header_font, body_font):
        """Draw comparison slide"""
        comparison_data = slide_data.get('comparison', {})
        
        # Two-column comparison
        col_width = (width - 3 * margin) // 2
        left_col = margin
        right_col = margin + col_width + margin
        
        # Left column
        if 'left' in comparison_data:
            draw.text((left_col, y_pos), comparison_data['left']['title'], 
                     fill=self.colors['primary'], font=header_font)
            y_pos += 60
            
            for item in comparison_data['left']['items']:
                draw.text((left_col, y_pos), f"• {item}", fill=self.colors['dark'], font=body_font)
                y_pos += 40
        
        # Right column
        y_pos = slide_data.get('start_y', y_pos)
        if 'right' in comparison_data:
            draw.text((right_col, y_pos), comparison_data['right']['title'], 
                     fill=self.colors['primary'], font=header_font)
            y_pos += 60
            
            for item in comparison_data['right']['items']:
                draw.text((right_col, y_pos), f"• {item}", fill=self.colors['dark'], font=body_font)
                y_pos += 40


# Utility functions for integration
def create_sample_legal_document_data() -> Dict[str, Any]:
    """Create sample legal document data for testing"""
    return {
        'title': 'Service Agreement Contract',
        'metadata': {
            'date': datetime.now().strftime('%B %d, %Y'),
            'case_number': 'SA-2024-001'
        },
        'sections': [
            {
                'title': '1. Parties',
                'content': 'This agreement is entered into between Company ABC and Client XYZ.'
            },
            {
                'title': '2. Services',
                'content': 'Company ABC agrees to provide legal consultation services to Client XYZ.'
            },
            {
                'title': '3. Terms',
                'content': 'This agreement shall be effective for a period of one year from the date of execution.'
            }
        ],
        'signature_blocks': [
            {
                'title': 'Company Representative',
                'name': 'John Smith, CEO',
                'date': datetime.now().strftime('%B %d, %Y')
            },
            {
                'title': 'Client Representative',
                'name': 'Jane Doe, Legal Counsel',
                'date': datetime.now().strftime('%B %d, %Y')
            }
        ],
        'watermark': 'CONFIDENTIAL'
    }


def create_sample_diagram_config() -> LegalDiagramConfig:
    """Create sample diagram configuration for testing"""
    return LegalDiagramConfig(
        diagram_type="legal_process",
        title="Legal Process Flow",
        description="Overview of the legal process from filing to resolution",
        nodes=[
            {'id': 'start', 'label': 'Start', 'position': (1, 5), 'type': 'start_end'},
            {'id': 'file', 'label': 'File Case', 'position': (3, 5), 'type': 'process'},
            {'id': 'serve', 'label': 'Serve Notice', 'position': (5, 5), 'type': 'process'},
            {'id': 'discovery', 'label': 'Discovery', 'position': (7, 5), 'type': 'process'},
            {'id': 'trial', 'label': 'Trial', 'position': (9, 5), 'type': 'process'},
            {'id': 'decision', 'label': 'Decision', 'position': (9, 3), 'type': 'decision'},
            {'id': 'end', 'label': 'End', 'position': (11, 3), 'type': 'start_end'}
        ],
        edges=[
            {'start': (1, 5), 'end': (3, 5), 'label': ''},
            {'start': (3, 5), 'end': (5, 5), 'label': ''},
            {'start': (5, 5), 'end': (7, 5), 'label': ''},
            {'start': (7, 5), 'end': (9, 5), 'label': ''},
            {'start': (9, 5), 'end': (9, 3), 'label': ''},
            {'start': (9, 3), 'end': (11, 3), 'label': ''}
        ],
        styling={'theme': 'professional'}
    )


def create_sample_infographic_data() -> Dict[str, Any]:
    """Create sample infographic data for testing"""
    return {
        'categories': {
            'Civil Cases': 45,
            'Criminal Cases': 30,
            'Family Law': 25,
            'Corporate Law': 20,
            'Constitutional': 15
        },
        'success_rate': 78,
        'timeline': {
            'dates': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'values': [10, 15, 12, 18, 22, 25]
        }
    }
