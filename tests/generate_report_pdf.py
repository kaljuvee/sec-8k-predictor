#!/usr/bin/env python3
"""
PDF Report Generator for SEC 8-K Prediction Analysis

This module generates professional PDF reports from markdown files using multiple
conversion methods for maximum compatibility and formatting quality.

Usage:
    python generate_report_pdf.py --input research_paper.md --output research_paper.pdf
    python generate_report_pdf.py --all  # Generate all reports
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import markdown
import weasyprint
from fpdf import FPDF
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportPDFGenerator:
    """Professional PDF generator for research reports and analysis documents."""
    
    def __init__(self, base_dir=None):
        """Initialize the PDF generator.
        
        Args:
            base_dir (str): Base directory containing markdown files. Defaults to test-data/
        """
        if base_dir is None:
            # Get the directory containing this script
            script_dir = Path(__file__).parent
            # Go up one level to project root, then to test-data
            self.base_dir = script_dir.parent / "test-data"
        else:
            self.base_dir = Path(base_dir)
            
        self.output_dir = self.base_dir / "pdf_reports"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"PDF Generator initialized with base_dir: {self.base_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        dependencies = {
            'weasyprint': self._check_weasyprint,
            'wkhtmltopdf': self._check_wkhtmltopdf,
            'pandoc': self._check_pandoc
        }
        
        available = {}
        for name, check_func in dependencies.items():
            try:
                available[name] = check_func()
                logger.info(f"✅ {name}: Available")
            except Exception as e:
                available[name] = False
                logger.warning(f"❌ {name}: Not available - {e}")
        
        return available
    
    def _check_weasyprint(self):
        """Check if WeasyPrint is available."""
        import weasyprint
        return True
    
    def _check_wkhtmltopdf(self):
        """Check if wkhtmltopdf is available."""
        result = subprocess.run(['wkhtmltopdf', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    
    def _check_pandoc(self):
        """Check if pandoc is available."""
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    
    def generate_with_weasyprint(self, markdown_file, output_file):
        """Generate PDF using WeasyPrint (best quality)."""
        logger.info(f"Generating PDF with WeasyPrint: {markdown_file} -> {output_file}")
        
        # Read markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc', 'footnotes']
        )
        
        # Add CSS styling for professional appearance
        css_style = """
        <style>
        @page {
            size: A4;
            margin: 2cm;
            @bottom-center {
                content: counter(page);
                font-size: 10pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: none;
        }
        
        h1 {
            font-size: 18pt;
            font-weight: bold;
            margin-top: 24pt;
            margin-bottom: 12pt;
            color: #1a1a1a;
            page-break-before: auto;
        }
        
        h2 {
            font-size: 14pt;
            font-weight: bold;
            margin-top: 18pt;
            margin-bottom: 9pt;
            color: #2c2c2c;
        }
        
        h3 {
            font-size: 12pt;
            font-weight: bold;
            margin-top: 12pt;
            margin-bottom: 6pt;
            color: #404040;
        }
        
        h4 {
            font-size: 11pt;
            font-weight: bold;
            margin-top: 9pt;
            margin-bottom: 4pt;
            color: #555;
        }
        
        p {
            margin-bottom: 6pt;
            text-align: justify;
        }
        
        ul, ol {
            margin-bottom: 6pt;
            padding-left: 20pt;
        }
        
        li {
            margin-bottom: 3pt;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 12pt 0;
            font-size: 10pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 6pt;
            text-align: left;
        }
        
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        
        code {
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f8f8f8;
            padding: 2pt;
            border-radius: 2pt;
        }
        
        pre {
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f8f8f8;
            padding: 8pt;
            border-radius: 4pt;
            overflow-x: auto;
            margin: 6pt 0;
        }
        
        blockquote {
            margin: 12pt 0;
            padding: 6pt 12pt;
            border-left: 3pt solid #ccc;
            background-color: #f9f9f9;
            font-style: italic;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        .no-break {
            page-break-inside: avoid;
        }
        
        strong {
            font-weight: bold;
        }
        
        em {
            font-style: italic;
        }
        
        hr {
            border: none;
            border-top: 1pt solid #ccc;
            margin: 18pt 0;
        }
        </style>
        """
        
        # Combine CSS and HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>SEC 8-K Prediction Analysis Report</title>
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        weasyprint.HTML(string=full_html).write_pdf(output_file)
        logger.info(f"✅ PDF generated successfully: {output_file}")
        return True
    
    def generate_with_pandoc(self, markdown_file, output_file):
        """Generate PDF using Pandoc (good compatibility)."""
        logger.info(f"Generating PDF with Pandoc: {markdown_file} -> {output_file}")
        
        cmd = [
            'pandoc',
            str(markdown_file),
            '-o', str(output_file),
            '--pdf-engine=xelatex',
            '--variable', 'geometry:margin=2cm',
            '--variable', 'fontsize=11pt',
            '--variable', 'linestretch=1.6',
            '--table-of-contents',
            '--number-sections'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ PDF generated successfully: {output_file}")
            return True
        else:
            logger.error(f"❌ Pandoc failed: {result.stderr}")
            return False
    
    def generate_with_fpdf(self, markdown_file, output_file):
        """Generate PDF using FPDF (fallback method)."""
        logger.info(f"Generating PDF with FPDF: {markdown_file} -> {output_file}")
        
        # Read markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple markdown to text conversion
        lines = content.split('\n')
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', size=12)
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                pdf.ln(5)
                continue
            
            # Handle headers
            if line.startswith('# '):
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, line[2:].strip(), ln=True)
                pdf.ln(5)
                pdf.set_font('Arial', size=12)
            elif line.startswith('## '):
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 8, line[3:].strip(), ln=True)
                pdf.ln(3)
                pdf.set_font('Arial', size=12)
            elif line.startswith('### '):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 6, line[4:].strip(), ln=True)
                pdf.ln(2)
                pdf.set_font('Arial', size=12)
            else:
                # Regular text
                try:
                    # Handle long lines by wrapping
                    if len(line) > 80:
                        words = line.split(' ')
                        current_line = ''
                        for word in words:
                            if len(current_line + word) < 80:
                                current_line += word + ' '
                            else:
                                if current_line:
                                    pdf.cell(0, 6, current_line.strip(), ln=True)
                                current_line = word + ' '
                        if current_line:
                            pdf.cell(0, 6, current_line.strip(), ln=True)
                    else:
                        pdf.cell(0, 6, line, ln=True)
                except UnicodeEncodeError:
                    # Skip lines with encoding issues
                    pdf.cell(0, 6, '[Content with special characters]', ln=True)
        
        pdf.output(output_file)
        logger.info(f"✅ PDF generated successfully: {output_file}")
        return True
    
    def generate_pdf(self, markdown_file, output_file=None, method='auto'):
        """Generate PDF from markdown file using the best available method.
        
        Args:
            markdown_file (str): Path to markdown file
            output_file (str): Output PDF path. If None, auto-generate from input name
            method (str): Generation method ('weasyprint', 'pandoc', 'fpdf', 'auto')
        
        Returns:
            bool: True if successful, False otherwise
        """
        markdown_path = Path(markdown_file)
        
        if not markdown_path.exists():
            logger.error(f"Markdown file not found: {markdown_file}")
            return False
        
        if output_file is None:
            output_file = self.output_dir / f"{markdown_path.stem}.pdf"
        else:
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check available methods
        available = self.check_dependencies()
        
        # Determine method to use
        if method == 'auto':
            if available.get('weasyprint', False):
                method = 'weasyprint'
            elif available.get('pandoc', False):
                method = 'pandoc'
            else:
                method = 'fpdf'
        
        # Generate PDF
        try:
            if method == 'weasyprint' and available.get('weasyprint', False):
                return self.generate_with_weasyprint(markdown_path, output_file)
            elif method == 'pandoc' and available.get('pandoc', False):
                return self.generate_with_pandoc(markdown_path, output_file)
            elif method == 'fpdf':
                return self.generate_with_fpdf(markdown_path, output_file)
            else:
                logger.error(f"Method '{method}' not available or not supported")
                return False
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False
    
    def generate_all_reports(self):
        """Generate PDFs for all markdown files in the test-data directory."""
        logger.info("Generating PDFs for all markdown reports...")
        
        markdown_files = [
            'research_paper.md',
            'executive_summary_report.md',
            'prediction_analysis_report.md',
            'README.md'
        ]
        
        results = {}
        
        for md_file in markdown_files:
            md_path = self.base_dir / md_file
            if md_path.exists():
                logger.info(f"Processing: {md_file}")
                success = self.generate_pdf(md_path)
                results[md_file] = success
                
                if success:
                    output_path = self.output_dir / f"{md_path.stem}.pdf"
                    file_size = output_path.stat().st_size / 1024  # KB
                    logger.info(f"✅ Generated: {output_path} ({file_size:.1f} KB)")
                else:
                    logger.error(f"❌ Failed to generate PDF for: {md_file}")
            else:
                logger.warning(f"⚠️  File not found: {md_file}")
                results[md_file] = False
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"\n📊 Summary: {successful}/{total} PDFs generated successfully")
        
        if successful > 0:
            logger.info(f"📁 Output directory: {self.output_dir}")
            logger.info("Generated files:")
            for pdf_file in self.output_dir.glob("*.pdf"):
                file_size = pdf_file.stat().st_size / 1024  # KB
                logger.info(f"  - {pdf_file.name} ({file_size:.1f} KB)")
        
        return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate professional PDF reports from markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_report_pdf.py --all
  python generate_report_pdf.py --input research_paper.md
  python generate_report_pdf.py --input research_paper.md --output report.pdf --method weasyprint
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Input markdown file path')
    parser.add_argument('--output', '-o', 
                       help='Output PDF file path (optional)')
    parser.add_argument('--method', '-m', 
                       choices=['weasyprint', 'pandoc', 'fpdf', 'auto'],
                       default='auto',
                       help='PDF generation method (default: auto)')
    parser.add_argument('--all', '-a', 
                       action='store_true',
                       help='Generate PDFs for all markdown files')
    parser.add_argument('--base-dir', '-d',
                       help='Base directory containing markdown files')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize generator
    generator = ReportPDFGenerator(base_dir=args.base_dir)
    
    if args.all:
        # Generate all reports
        results = generator.generate_all_reports()
        success_count = sum(results.values())
        if success_count > 0:
            print(f"\n🎉 Successfully generated {success_count} PDF reports!")
            print(f"📁 Check the output directory: {generator.output_dir}")
        else:
            print("\n❌ No PDFs were generated successfully.")
            sys.exit(1)
    
    elif args.input:
        # Generate single report
        success = generator.generate_pdf(args.input, args.output, args.method)
        if success:
            output_file = args.output or generator.output_dir / f"{Path(args.input).stem}.pdf"
            print(f"\n🎉 PDF generated successfully: {output_file}")
        else:
            print(f"\n❌ Failed to generate PDF from: {args.input}")
            sys.exit(1)
    
    else:
        parser.print_help()
        print("\nError: Please specify --input file or --all flag")
        sys.exit(1)

if __name__ == "__main__":
    main()

