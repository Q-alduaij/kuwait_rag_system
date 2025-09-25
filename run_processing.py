import os
import json
import time
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from datetime import datetime
from config.settings import settings
from models.schemas import DocumentChunk, ProcessingResult, ContentType
from data_processing.chunkers import DynamicChunker
from data_processing.file_handlers import FileHandlerFactory, LANGCHAIN_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOGS_DIR, 'processing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    def __init__(self):
        self.chunker = DynamicChunker()
        self.file_handler_factory = FileHandlerFactory()
        self.processed_chunks: List[DocumentChunk] = []
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Create necessary directories
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        os.makedirs(settings.LOGS_DIR, exist_ok=True)
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        logger.info("✅ EnhancedDocumentProcessor initialized")
        if LANGCHAIN_AVAILABLE:
            logger.info("✅ LangChain document loaders available")
        else:
            logger.warning("⚠️ LangChain not available, using fallback handlers")

    def determine_content_type(self, filename: str, filepath: str) -> ContentType:
        """Enhanced content type detection with better pattern matching"""
        filename_lower = filename.lower()
        file_content = filename_lower + " " + filepath.lower()
        
        # Quran detection (high priority)
        if any(keyword in file_content for keyword in ['quran', 'قرآن', 'قران', 'مصحف']):
            return ContentType.QURAN
        
        # Tafsir detection
        if any(keyword in file_content for keyword in ['tafsir', 'تفسير', 'شرح']):
            return ContentType.TAFSIR
        
        # Hadith detection
        if any(keyword in file_content for keyword in ['hadith', 'حديث', 'أحاديث', 'صحيح', 'سنن']):
            return ContentType.HADITH
        
        # Islamic history detection
        if any(keyword in file_content for keyword in ['islamic history', 'تاريخ إسلامي', 'سيرة', 'فقه']):
            return ContentType.ISLAMIC_HISTORY
        
        # Legal detection
        if any(keyword in file_content for keyword in ['law', 'قانون', 'دستور', 'مادة', 'تشريع', 'كويتي']):
            return ContentType.LEGAL
        
        # Kuwaiti dialect detection
        if any(keyword in file_content for keyword in ['لهجة', 'كويتي', 'dialect', 'عامية']):
            return ContentType.KUWAITI_DIALECT
        
        # History detection
        if any(keyword in file_content for keyword in ['history', 'تاريخ', 'تاريخ الكويت']):
            return ContentType.HISTORY
        
        # Culture detection
        if any(keyword in file_content for keyword in ['culture', 'ثقافة', 'تراث', 'عادات']):
            return ContentType.CULTURE
        
        return ContentType.MIXED

    def validate_file(self, filepath: str) -> bool:
        """Validate file before processing"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        if not os.path.isfile(filepath):
            logger.error(f"Path is not a file: {filepath}")
            return False
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        if file_size > settings.MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {filepath} ({file_size:.2f}MB)")
            return False
        
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext not in settings.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {filepath}")
            return False
        
        return True

    def process_file(self, filepath: str) -> ProcessingResult:
        """Process a single file with enhanced error handling"""
        try:
            start_time = time.time()
            
            # Validate file
            if not self.validate_file(filepath):
                return ProcessingResult(
                    success=False,
                    error_message="File validation failed"
                )
            
            # Determine content type
            filename = os.path.basename(filepath)
            content_type = self.determine_content_type(filename, filepath)
            logger.info(f"📄 Processing {filename} as {content_type.value}")
            
            # Extract text using appropriate handler
            handler = self.file_handler_factory.get_handler(filepath)
            text_content = handler.extract_text(filepath)
            
            if not text_content or not text_content.strip():
                return ProcessingResult(
                    success=False,
                    error_message="No text content extracted or empty content"
                )
            
            logger.info(f"📝 Extracted {len(text_content)} characters from {filename}")
            
            # Chunk document based on content type
            chunks = self.chunker.chunk_document(text_content, filename, content_type)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    error_message="No chunks generated from content"
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Successfully processed {filename}: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                chunks=chunks,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing {filepath}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def process_directory(self, directory_path: str):
        """Process all files in a directory with progress tracking"""
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return
        
        self.processing_stats['start_time'] = datetime.now()
        logger.info(f"🚀 Starting processing of directory: {directory_path}")
        
        # Collect all supported files
        supported_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in settings.SUPPORTED_EXTENSIONS):
                    supported_files.append(os.path.join(root, file))
        
        self.processing_stats['total_files'] = len(supported_files)
        logger.info(f"📁 Found {len(supported_files)} supported files")
        
        # Process files with progress bar
        for filepath in tqdm(supported_files, desc="Processing files"):
            result = self.process_file(filepath)
            
            if result.success:
                self.processed_chunks.extend(result.chunks)
                self.processing_stats['successful_files'] += 1
                self.processing_stats['total_chunks'] += len(result.chunks)
            else:
                self.processing_stats['failed_files'] += 1
                logger.error(f"Failed to process {os.path.basename(filepath)}: {result.error_message}")
        
        self.processing_stats['end_time'] = datetime.now()
        
        # Log summary
        self._log_processing_summary()

    def _log_processing_summary(self):
        """Log detailed processing summary"""
        duration = self.processing_stats['end_time'] - self.processing_stats['start_time']
        
        summary = f"""
📊 PROCESSING SUMMARY
────────────────────
• Total files: {self.processing_stats['total_files']}
• Successful: {self.processing_stats['successful_files']} ✅
• Failed: {self.processing_stats['failed_files']} ❌
• Total chunks: {self.processing_stats['total_chunks']}
• Processing time: {duration}
• Chunks per second: {self.processing_stats['total_chunks'] / max(duration.total_seconds(), 1):.2f}
"""
        logger.info(summary)

    def save_chunks(self, output_path: str = None):
        """Save processed chunks to JSONL file with backup"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(settings.PROCESSED_DIR, f"chunks_{timestamp}.jsonl")
        
        try:
            # Create backup if file exists
            if os.path.exists(output_path):
                backup_path = output_path + ".backup"
                os.rename(output_path, backup_path)
                logger.info(f"📦 Created backup: {backup_path}")
            
            # Save chunks
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in self.processed_chunks:
                    chunk_dict = {
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                    f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Saved {len(self.processed_chunks)} chunks to {output_path}")
            
            # Save processing stats
            stats_path = output_path.replace('.jsonl', '_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                stats = self.processing_stats.copy()
                stats['duration'] = str(stats['end_time'] - stats['start_time'])
                stats['start_time'] = stats['start_time'].isoformat()
                stats['end_time'] = stats['end_time'].isoformat()
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📈 Saved processing stats to {stats_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save chunks: {str(e)}")
            raise

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed chunks"""
        if not self.processed_chunks:
            return {}
        
        stats = {
            'total_chunks': len(self.processed_chunks),
            'content_type_distribution': {},
            'sensitivity_distribution': {},
            'language_variant_distribution': {},
            'average_chunk_length': 0,
            'total_characters': 0
        }
        
        total_chars = 0
        for chunk in self.processed_chunks:
            content_type = chunk.metadata.get('source_type', 'unknown')
            sensitivity = chunk.metadata.get('sensitivity_level', 'unknown')
            language = chunk.metadata.get('language_variant', 'unknown')
            
            # Update distributions
            stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
            stats['sensitivity_distribution'][sensitivity] = stats['sensitivity_distribution'].get(sensitivity, 0) + 1
            stats['language_variant_distribution'][language] = stats['language_variant_distribution'].get(language, 0) + 1
            
            # Calculate length statistics
            chunk_length = len(chunk.content)
            total_chars += chunk_length
        
        stats['average_chunk_length'] = total_chars / len(self.processed_chunks) if self.processed_chunks else 0
        stats['total_characters'] = total_chars
        
        return stats

def main():
    """Main processing function"""
    try:
        processor = EnhancedDocumentProcessor()
        
        # Process data directory
        logger.info("🚀 Starting Kuwait RAG document processing...")
        processor.process_directory(settings.DATA_DIR)
        
        # Save results
        processor.save_chunks()
        
        # Print statistics
        stats = processor.get_chunk_statistics()
        logger.info("📊 Chunk Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("🎉 Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Fatal error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()