"""
Data Processor for FIAP Tech Challenge Phase 3
Handles data cleaning, validation and formatting for fine-tuning
"""

import os
import json
import re
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from config import Config


class DataProcessor:
    """Processes and cleans dataset for fine-tuning."""

    def __init__(self, input_path: str, output_path: str,
                 config: Optional[Config] = Config()):
        """
        Initialize processor with input and output paths.

        Args:
            input_path: Path to raw dataset
            output_path: Path for processed dataset
            config: Processing configuration object
        """
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.stats = {
            "total_processed": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "empty_content": 0,
            "empty_title": 0,
            "processing_errors": 0
        }

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing problematic characters.

        Args:
            text: Input text to clean

        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        text = ''.join(char for char in text
                       if ord(char) >= 32 or char in '\n\t')

        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate if record is suitable for fine-tuning.

        Args:
            record: Record to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(record, dict):
            return False

        for field in self.config.required_fields:
            if field not in record:
                return False

        title = self.clean_text(record.get('title', ''))
        if self._empty_title(title):
            self.stats["empty_title"] += 1
            return False

        # Check content - reject if empty
        content = self.clean_text(record.get('content', ''))
        if self._empty_content(content):
            self.stats["empty_content"] += 1
            return False  # Reject records with empty content
            
        return True

    def _empty_title(self, title: str) -> bool:
        """
        Check if title is empty.

        Args:
            title: Title to check
        """
        return len(title) < self.config.min_title_length

    def _empty_content(self, content: str) -> bool:
        """
        Check if content is empty.
        """
        return len(content) < self.config.min_content_length

    def format_to_alpaca(self, record: Dict[str, Any]) -> Optional[Dict]:
        """
        Convert record to Alpaca format for fine-tuning.

        Args:
            record: Input record

        Returns:
            dict: Alpaca formatted record or None if invalid
        """
        try:
            title = self.clean_text(record.get('title', ''))
            content = self.clean_text(record.get('content', ''))

            if self._empty_title(title):
                return None
            
            if self._empty_content(content):
                return None

            # Create Alpaca format
            instruction = ("Generate a detailed description for the "
                          "following item.")
            input_text = title
            output_text = content

            # Create the complete Alpaca template
            template = ("Below is an instruction that describes a task, "
                       "paired with an input that provides further context. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")
            alpaca_template = template

            alpaca_record = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "text": alpaca_template.format(instruction, input_text,
                                              output_text)
            }

            # Add original fields for reference
            if 'uid' in record:
                alpaca_record['uid'] = record['uid']

            return alpaca_record

        except Exception as e:
            print(f"Error formatting record: {e}")
            return None

    def process_chunk(self, chunk: List[tuple], output_file) -> tuple:
        """
        Process a chunk of records.

        Args:
            chunk: List of (line_num, record) tuples
            output_file: Output file handle

        Returns:
            tuple: (valid_count, invalid_count)
        """
        valid_count = 0
        invalid_count = 0

        for line_num, record in chunk:
            try:
                # Validate record
                if not self.validate_record(record):
                    invalid_count += 1
                    continue

                # Format to Alpaca
                alpaca_record = self.format_to_alpaca(record)

                if alpaca_record:
                    # Write formatted record
                    output_file.write(json.dumps(alpaca_record,
                                                 ensure_ascii=False) + '\n')
                    output_file.flush()
                    valid_count += 1
                else:
                    invalid_count += 1

            except Exception as e:
                self.stats["processing_errors"] += 1
                invalid_count += 1
                if self.stats["processing_errors"] <= 5:
                    print(f"Error processing line {line_num}: {e}")

        return valid_count, invalid_count

    def process_dataset(self) -> bool:
        """
        Process entire dataset in chunks.

        Args:
            chunk_size: Size of chunks to process

        Returns:
            bool: True if successful
        """
        print(f"Processing dataset in chunks of {self.config.chunk_size}...")
        print("This will filter out records with empty content")

        # Remove output file if exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        try:
            with open(self.input_path, 'r', encoding='utf-8',
                      errors='replace') as infile, \
                 open(self.output_path, 'w', encoding='utf-8') as outfile:

                chunk = []
                chunk_count = 0

                # Create progress bar
                pbar = tqdm(desc="Processing", unit=" lines")

                for line_num, line in enumerate(infile):
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        record = json.loads(line)
                        chunk.append((line_num + 1, record))

                        # Process chunk when reaching size
                        if len(chunk) >= self.config.chunk_size:
                            valid, invalid = self.process_chunk(chunk, outfile)
                            self.stats["valid_records"] += valid
                            self.stats["invalid_records"] += invalid
                            self.stats["total_processed"] += len(chunk)
                            chunk_count += 1

                            # Update progress
                            rate_calc = (self.stats['valid_records'] + self.stats['invalid_records'])
                            if rate_calc > 0:
                                rate_calc_rate = self.stats['valid_records']/rate_calc*100
                                rate = f"{rate_calc_rate:.1f}%"
                            else:
                                rate = "0%"
                            
                            pbar.set_postfix({
                                'Valid': self.stats["valid_records"],
                                'Invalid': self.stats["invalid_records"],
                                'Rate': rate
                            })
                            pbar.update(len(chunk))

                            chunk = []  # Clear memory

                    except json.JSONDecodeError:
                        self.stats["processing_errors"] += 1
                    except Exception as e:
                        self.stats["processing_errors"] += 1

                # Process last chunk
                if chunk:
                    valid, invalid = self.process_chunk(chunk, outfile)
                    self.stats["valid_records"] += valid
                    self.stats["invalid_records"] += invalid
                    self.stats["total_processed"] += len(chunk)
                    pbar.update(len(chunk))

                pbar.close()

            self._print_processing_summary()
            return True

        except Exception as e:
            print(f"Error processing dataset: {e}")
            return False

    def _print_processing_summary(self) -> None:
        """Print summary of processing results."""
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total processed: {self.stats['total_processed']:,}")
        print(f"Valid records: {self.stats['valid_records']:,}")
        print(f"Invalid records: {self.stats['invalid_records']:,}")
        print(f"Empty content handled: {self.stats['empty_content']:,}")
        print(f"Empty titles: {self.stats['empty_title']:,}")
        print(f"Processing errors: {self.stats['processing_errors']:,}")

        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['valid_records'] /
                            self.stats['total_processed'] * 100)
            print(f"Success rate: {success_rate:.1f}%")

        if os.path.exists(self.output_path):
            size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            print(f"Output file: {self.output_path}")
            print(f"Output size: {size_mb:.1f} MB")


def process_dataset(input_path: str, output_path: str,
                    config: Optional[Config] = None) -> bool:
    """
    Convenience function to process dataset.

    Args:
        input_path: Input dataset path
        output_path: Output dataset path
        config: Processing configuration

    Returns:
        bool: True if successful
    """
    if config is None:
        config = Config()

    processor = DataProcessor(input_path, output_path, config)
    return processor.process_dataset()
