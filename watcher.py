"""File watcher with queue and worker threads for Excel processing."""

import logging
import queue
import signal
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import (
    ERROR_DIR,
    COMPLETED_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    NUM_WORKERS,
    PROCESS_DIR,
    SOURCE_DIR,
    TARGET_DIR,
)
from core import extract, read_excel
from core.data_cleaner import clean_data
from core.target_schema import TABLE_SPEC
from core.writer import write_excel
from core.recipe_engine import apply_recipe, generate_recipe, validate_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('excel_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExcelHandler(FileSystemEventHandler):
    """Handles file system events for Excel files."""

    def __init__(self, file_queue: queue.Queue):
        self.file_queue = file_queue

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.xlsx'):
            logger.info(f"New Excel file detected: {event.src_path}")
            # Wait a moment for file to be fully written
            time.sleep(0.5)
            self._enqueue_file(Path(event.src_path))

    def _enqueue_file(self, filepath: Path):
        """Add file to processing queue."""
        try:
            # Move to process directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # process_name = f"{filepath.stem}_{timestamp}{filepath.suffix}" # TODO: rename
            process_name = f"{filepath.stem}{filepath.suffix}"
            process_path = PROCESS_DIR / process_name

            # Move file
            filepath.rename(process_path)

            # Add to queue
            self.file_queue.put(process_path)
            logger.info(f"Enqueued file: {process_path}")

        except Exception as e:
            logger.error(f"Error enqueuing file {filepath}: {e}")


def worker_thread(
        file_queue: queue.Queue,
        worker_id: int,
        shutdown_event: threading.Event
):
    """Worker thread that processes files from the queue."""
    logger.info(f"Worker {worker_id} started")

    while not shutdown_event.is_set():
        try:
            # Wait for file with timeout
            try:
                filepath = file_queue.get(timeout=1.0)
                _input_file_path = Path(filepath)
            except queue.Empty:
                continue

            logger.info(f"Worker {worker_id} processing: {_input_file_path.name}")

            try:

                # Step 1: Detect structure
                target_columns = TABLE_SPEC.column_names
                logger.info(f"Worker {worker_id} detecting structure...")
                structure =extract(filepath, target_columns=target_columns)
                logger.info(
                    f"Worker {worker_id} detected structure: {structure.boundary.data_row_count} rows, {len(structure.dataframe.columns)} columns")
                # print(structure)

                df_structured = structure.dataframe

                # Strip whitespace from column names to handle trailing spaces
                df_structured.columns = df_structured.columns.str.strip()
                
                # Debug: Show columns and their dtypes
                print("\n=== DataFrame Structure ===")
                print(f"Shape: {df_structured.shape}")
                print("\nColumns and dtypes:")
                for col in df_structured.columns:
                    print(f"  {col}: {df_structured[col].dtype}")
                print("========================\n")

                # Step 2: Generate recipe
                logger.info(f"Worker {worker_id} generating recipe...")
                recipe = generate_recipe(
                    df_structured,
                    structure,
                    target_columns,
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_MODEL
                )
                logger.info(f"Worker {worker_id} generated recipe with {len(recipe.transformations)} transformations")
                print(recipe)

                # Step 3: Apply recipe
                logger.info(f"Worker {worker_id} applying recipe...")
                df = apply_recipe(df_structured, recipe, target_columns)
                logger.info(f"Worker {worker_id} applied recipe, got {len(df)} rows")

                val_warnings = validate_output(df_structured, df, recipe)
                for w in val_warnings:
                    logger.warning(f"Worker {worker_id} validation: {w}")

                # Step 3.1 Data Cleanup
                logger.info(f"Worker {worker_id} Data Cleansing for file {_input_file_path.name}...")
                df, _ = clean_data(df)

                # Step 4: Write output
                output_path = TARGET_DIR / f"{_input_file_path.stem}.xlsx"
                write_excel(df, output_path)
                logger.info(f"Worker {worker_id} wrote output to: {output_path}")

                # Step 6: Move to completed
                completed_path = COMPLETED_DIR / filepath.name
                filepath.rename(completed_path)
                logger.info(f"Worker {worker_id} moved processed file to: {completed_path}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {filepath}: {e}")
                logger.error(f"Worker {worker_id} traceback: {traceback.format_exc()}")
                # Move to error directory on error
                try:
                    archive_path = ERROR_DIR / filepath.name
                    filepath.rename(archive_path)
                    logger.info(f"Worker {worker_id} moved failed file to archive: {archive_path}")
                except Exception as archive_error:
                    logger.error(f"Worker {worker_id} error archiving file: {archive_error}")

            finally:
                file_queue.task_done()

        except Exception as e:
            logger.error(f"Worker {worker_id} unexpected error: {e}")

    logger.info(f"Worker {worker_id} shutting down")


def enqueue_existing_files(file_queue: queue.Queue):
    """Enqueue any existing Excel files in source directory."""
    for filepath in SOURCE_DIR.glob("*.xlsx"):
        logger.info(f"Enqueuing existing file: {filepath}")
        # Move to process directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        process_path = PROCESS_DIR / process_name

        try:
            filepath.rename(process_path)
            file_queue.put(process_path)
        except Exception as e:
            logger.error(f"Error enqueuing existing file {filepath}: {e}")

    for p_filepath in PROCESS_DIR.glob("*.xlsx"):
        logger.info(f"Enqueuing existing file under processing folder: {p_filepath}")
        try:
            file_queue.put(p_filepath)
        except Exception as e:
            logger.error(f"Error enqueuing existing file {p_filepath}: {e}")


def main():
    """Main watcher function."""
    logger.info("Starting Excel ingestion watcher...")

    # Create shutdown event
    shutdown_event = threading.Event()

    # Create file queue
    file_queue = queue.Queue()

    # Enqueue existing files
    enqueue_existing_files(file_queue)

    # Start worker threads
    workers = []
    for i in range(NUM_WORKERS):
        worker = threading.Thread(
            target=worker_thread,
            args=(file_queue, i + 1, shutdown_event),
            name=f"Worker-{i + 1}"
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Setup file system watcher
    event_handler = ExcelHandler(file_queue)
    observer = Observer()
    observer.schedule(event_handler, str(SOURCE_DIR), recursive=False)
    observer.start()

    logger.info(f"Watcher started on {SOURCE_DIR} with {NUM_WORKERS} workers")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Main loop
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        shutdown_event.set()

    # Stop observer
    observer.stop()
    observer.join()

    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=5)

    logger.info("Watcher shutdown complete")


if __name__ == "__main__":
    main()
