"""
Dataset Analyzer for FIAP Tech Challenge Phase 3
Handles dataset analysis and statistics without loading everything into memory
"""

import os
import json
import psutil
from typing import Dict, Any


class DatasetAnalyzer:
    """Analyzes dataset structure and provides statistics safely."""

    def __init__(self, file_path: str):
        """
        Initialize analyzer with dataset file path.

        Args:
            file_path: Path to the dataset file
        """
        self.file_path = file_path
        self.file_exists = os.path.exists(file_path)
        self.analysis_results = {}

    def check_memory_usage(self) -> float:
        """
        Monitor system memory usage to prevent crashes.

        Returns:
            float: Memory usage percentage
        """
        memory = psutil.virtual_memory()
        usage_gb = memory.used / (1024**3)
        total_gb = memory.total / (1024**3)
        print(f"Memory: {memory.percent:.1f}% used "
              f"({usage_gb:.1f}GB/{total_gb:.1f}GB)")

        if memory.percent > 80:
            print("WARNING: High memory usage! Consider reducing chunk_size")
        elif memory.percent > 90:
            print("CRITICAL: Very high memory usage! Risk of crash!")

        return memory.percent

    def count_lines_efficiently(self) -> int:
        """
        Count lines in file efficiently without loading everything.

        Returns:
            int: Number of lines in the file
        """
        if not self.file_exists:
            print(f"ERROR: File not found: {self.file_path}")
            return 0

        print(f"Counting lines in {self.file_path}...")

        line_count = 0
        try:
            with open(self.file_path, 'r', encoding='utf-8',
                      errors='replace') as f:
                for line in f:
                    line_count += 1

            print(f"Total lines: {line_count:,}")
            return line_count

        except Exception as e:
            print(f"Error counting lines: {e}")
            return 0

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get comprehensive file information.

        Returns:
            dict: File information including size, lines, etc.
        """
        if not self.file_exists:
            return {"exists": False, "error": "File not found"}

        try:
            size_bytes = os.path.getsize(self.file_path)
            size_mb = size_bytes / (1024 * 1024)
            line_count = self.count_lines_efficiently()

            return {
                "exists": True,
                "path": self.file_path,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "total_lines": line_count,
                "avg_bytes_per_line": size_bytes / line_count if line_count > 0 else 0
            }

        except Exception as e:
            return {"exists": True, "error": str(e)}

    def analyze_structure(self, sample_size: int = 50) -> Dict[str, Any]:
        """
        Analyze dataset structure without loading everything into memory.

        Args:
            sample_size: Number of records to sample for analysis

        Returns:
            dict: Analysis results including fields, examples, errors
        """
        if not self.file_exists:
            return {"error": "File not found"}

        print(f"Analyzing dataset structure ({sample_size} samples)...")

        sample_records = []
        field_counts = {}
        error_count = 0
        field_types = {}
        field_lengths = {}

        try:
            with open(self.file_path, 'r', encoding='utf-8',
                      errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break

                    try:
                        line = line.strip()
                        if not line:
                            continue

                        record = json.loads(line)
                        sample_records.append(record)

                        # Count fields and analyze types
                        for field, value in record.items():
                            # Count field frequency
                            field_counts[field] = field_counts.get(field, 0) + 1

                            # Track field types
                            field_type = type(value).__name__
                            if field not in field_types:
                                field_types[field] = {}
                            field_types[field][field_type] = (
                                field_types[field].get(field_type, 0) + 1
                            )

                            # Track field lengths for strings
                            if isinstance(value, str):
                                if field not in field_lengths:
                                    field_lengths[field] = []
                                field_lengths[field].append(len(value))

                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 3:
                            print(f"JSON Error line {i+1}: {str(e)[:100]}...")

        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

        # Calculate statistics
        analysis_results = {
            "sample_size": len(sample_records),
            "total_errors": error_count,
            "fields_found": list(field_counts.keys()),
            "field_frequency": {
                field: {
                    "count": count,
                    "percentage": (count / len(sample_records) * 100)
                    if len(sample_records) > 0 else 0
                }
                for field, count in field_counts.items()
            },
            "field_types": field_types,
            "field_length_stats": {}
        }

        # Calculate length statistics for string fields
        for field, lengths in field_lengths.items():
            if lengths:
                analysis_results["field_length_stats"][field] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "avg": sum(lengths) / len(lengths),
                    "samples": len(lengths)
                }

        # Store example record
        if sample_records:
            analysis_results["example_record"] = sample_records[0]

        self.analysis_results = analysis_results
        return analysis_results

    def print_analysis_summary(self) -> None:
        """Print a formatted summary of the analysis results."""
        if not self.analysis_results:
            print("No analysis results available. "
                  "Run analyze_structure() first.")
            return

        results = self.analysis_results
        print("\n=== DATASET ANALYSIS SUMMARY ===")
        print(f"Sample size: {results['sample_size']}")
        print(f"Parse errors: {results['total_errors']}")
        print(f"Fields found: {len(results['fields_found'])}")

        print("\nField frequency:")
        for field, stats in results['field_frequency'].items():
            print(f"  - {field}: {stats['count']}/{results['sample_size']} "
                  f"({stats['percentage']:.1f}%)")

        if results.get('field_length_stats'):
            print("\nString field lengths:")
            for field, stats in results['field_length_stats'].items():
                print(f"  - {field}: avg={stats['avg']:.0f}, "
                      f"min={stats['min']}, max={stats['max']}")

        if results.get('example_record'):
            print("\nExample record structure:")
            for key, value in results['example_record'].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

    def get_recommended_chunk_size(self) -> int:
        """
        Get recommended chunk size based on file size and memory.

        Returns:
            int: Recommended chunk size
        """
        file_info = self.get_file_info()
        memory_percent = self.check_memory_usage()

        if not file_info.get("exists"):
            return 500  # Default

        size_mb = file_info["size_mb"]
        total_lines = file_info["total_lines"]

        # Base recommendation on file size and memory usage
        if size_mb > 1000 or memory_percent > 70:
            recommended = 100
        elif size_mb > 500 or memory_percent > 60:
            recommended = 200
        elif size_mb > 100:
            recommended = 500
        else:
            recommended = 1000

        # Adjust based on total lines
        if total_lines > 5000000:  # 5M+ lines
            recommended = min(recommended, 100)
        elif total_lines > 1000000:  # 1M+ lines
            recommended = min(recommended, 300)

        print(f"Recommended chunk size: {recommended}")
        return recommended

    def validate_dataset_format(self) -> bool:
        """
        Validate that the dataset has the expected format.

        Returns:
            bool: True if format is valid for fine-tuning
        """
        if not self.file_exists:
            print("ERROR: File does not exist")
            return False

        try:
            with open(self.file_path, 'r', encoding='utf-8',
                      errors='replace') as f:
                first_line = f.readline().strip()
                if first_line.startswith('{'):
                    print("Format: JSON Lines (JSONL) - Compatible")
                    return True
                elif first_line.startswith('['):
                    print("Format: JSON Array - Needs conversion")
                    return True
                else:
                    print("Format: Unknown - Not compatible")
                    print(f"First line: {first_line[:50]}...")
                    return False
        except Exception as e:
            print(f"Error validating format: {e}")
            return False


def analyze_dataset(file_path: str, sample_size: int = 50) -> DatasetAnalyzer:
    """
    Convenience function to create analyzer and run complete analysis.

    Args:
        file_path: Path to dataset file
        sample_size: Number of records to sample

    Returns:
        DatasetAnalyzer: Analyzer instance with results
    """
    analyzer = DatasetAnalyzer(file_path)
    analyzer.check_memory_usage()
    analyzer.get_file_info()
    analyzer.analyze_structure(sample_size)
    analyzer.validate_dataset_format()
    analyzer.print_analysis_summary()
    return analyzer
