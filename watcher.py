"""File watcher with queue and worker threads for Excel processing."""

import logging
import logging.handlers
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
    OUTPUT_WORKERS,
    PROCESS_DIR,
    SOURCE_DIR,
    TARGET_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
)
from core import extract, read_excel
from core.data_cleaner import clean_data
from core.target_schema import TABLE_SPEC
from core.writer import write_excel
from core.recipe_engine import apply_recipe, generate_recipe, validate_output


def safe_move_file(src: Path, dst: Path) -> None:
    """Safely move file across platforms, handling cross-device moves."""
    try:
        src.rename(dst)
    except OSError as e:
        # Handle cross-device moves (e.g., different drives on Windows)
        if "cross-device" in str(e).lower() or (hasattr(e, 'winerror') and e.winerror == 17):
            import shutil
            shutil.copy2(src, dst)
            src.unlink()
        else:
            raise


# Configure logging with daily rotation
def setup_logging():
    """Setup logging with daily log files."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)

    # Generate log filename with current date
    log_filename = LOGS_DIR / f"excel_ingestion_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),  # Append mode
            logging.StreamHandler()
        ]
    )


setup_logging()
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
            time.sleep(5)
            self._enqueue_file(Path(event.src_path))

    def _enqueue_file(self, filepath: Path):
        """Add file to processing queue."""
        try:
            # Move to process directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            process_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            # process_name = f"{filepath.stem}{filepath.suffix}"
            process_path = PROCESS_DIR / process_name

            # Move file (cross-platform compatible)
            safe_move_file(filepath, process_path)

            # Add to queue
            self.file_queue.put(process_path)
            logger.info(f"Enqueued file: {process_path}")

        except Exception as e:
            logger.error(f"Error enqueuing file {filepath}: {e}")


def worker_thread(
        file_queue: queue.Queue,
        output_queue: queue.Queue,
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
                structure = extract(filepath, target_columns=target_columns)
                logger.info(
                    f"Worker {worker_id} detected structure: {structure.boundary.data_row_count} rows, {len(structure.dataframe.columns)} columns")

                df_structured = structure.dataframe

                # Strip whitespace from column names to handle trailing spaces
                df_structured.columns = df_structured.columns.str.strip()

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
                logger.debug(recipe)

                # Step 3: Apply recipe
                logger.info(f"Worker {worker_id} applying recipe...")
                df = apply_recipe(df_structured, recipe, target_columns)
                logger.info(f"Worker {worker_id} applied recipe, got {len(df)} rows")

                val_warnings = validate_output(df_structured, df, recipe)
                for w in val_warnings:
                    logger.warning(f"Worker {worker_id} validation: {w}")

                # Step 4: Write output to target directory
                output_path = TARGET_DIR / f"{_input_file_path.stem}.xlsx"
                write_excel(df, output_path, apply_formatting=False)
                logger.info(f"Worker {worker_id} wrote output to: {output_path}")

                # Step 5: Enqueue to output queue for data cleaning and final output
                output_task = {
                    'source_file': str(_input_file_path),
                    'target_file': str(output_path),
                    'worker_id': worker_id
                }
                output_queue.put(output_task)
                logger.info(f"Worker {worker_id} enqueued file for output processing: {output_path}")

                # Step 6: Move to completed
                completed_path = COMPLETED_DIR / filepath.name
                safe_move_file(filepath, completed_path)
                logger.info(f"Worker {worker_id} moved processed file to: {completed_path}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {filepath}: {e}")
                logger.error(f"Worker {worker_id} traceback: {traceback.format_exc()}")
                # Move to error directory on error
                try:
                    archive_path = ERROR_DIR / filepath.name
                    safe_move_file(filepath, archive_path)
                    logger.info(f"Worker {worker_id} moved failed file to archive: {archive_path}")
                except Exception as archive_error:
                    logger.error(f"Worker {worker_id} error archiving file: {archive_error}")

            finally:
                file_queue.task_done()

        except Exception as e:
            logger.error(f"Worker {worker_id} unexpected error: {e}")

    logger.info(f"Worker {worker_id} shutting down")


def output_worker_thread(
        output_queue: queue.Queue,
        worker_id: int,
        shutdown_event: threading.Event
):
    """Output worker thread that processes files from the output queue."""
    logger.info(f"Output Worker {worker_id} started")

    while not shutdown_event.is_set():
        try:
            # Wait for task with timeout
            try:
                output_task = output_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            logger.info(f"Output Worker {worker_id} processing: {Path(output_task['target_file']).name}")

            try:
                # Step 1: Read the target file
                logger.info(f"Output Worker {worker_id} reading target file: {output_task['target_file']}")
                df = read_excel(output_task['target_file'], header=0).df

                # Step 2: Apply data cleaning
                logger.info(
                    f"Output Worker {worker_id} Data Cleansing for file {Path(output_task['source_file']).name}...")
                cleaned_df, _ = clean_data(df)

                # Step 3: Write to output directory
                output_final_path = OUTPUT_DIR / f"{Path(output_task['source_file']).stem}_output.xlsx"  # Todo: remove suffix output
                write_excel(cleaned_df, output_final_path)
                logger.info(f"Output Worker {worker_id} wrote cleaned output to: {output_final_path}")

            except Exception as e:
                logger.error(f"Output Worker {worker_id} error processing {output_task['target_file']}: {e}")
                logger.error(f"Output Worker {worker_id} traceback: {traceback.format_exc()}")

            finally:
                output_queue.task_done()

        except Exception as e:
            logger.error(f"Output Worker {worker_id} unexpected error: {e}")

    logger.info(f"Output Worker {worker_id} shutting down")


def enqueue_existing_files(file_queue: queue.Queue):
    """Enqueue any existing Excel files in source directory."""
    for filepath in SOURCE_DIR.glob("*.xlsx"):
        logger.info(f"Moving existing file: {filepath} to Process folder")
        # Move to process directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        process_path = PROCESS_DIR / process_name

        try:
            safe_move_file(filepath, process_path)
            # file_queue.put(process_path)
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

    # Create file queues
    file_queue = queue.Queue()
    output_queue = queue.Queue()

    # Enqueue existing files
    enqueue_existing_files(file_queue)

    # Start main worker threads
    workers = []
    for i in range(NUM_WORKERS):
        worker = threading.Thread(
            target=worker_thread,
            args=(file_queue, output_queue, i + 1, shutdown_event),
            name=f"Worker-{i + 1}"
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Start output worker threads
    output_workers = []
    for i in range(OUTPUT_WORKERS):
        output_worker = threading.Thread(
            target=output_worker_thread,
            args=(output_queue, i + 1, shutdown_event),
            name=f"OutputWorker-{i + 1}"
        )
        output_worker.daemon = True
        output_worker.start()
        output_workers.append(output_worker)

    # Setup file system watcher
    event_handler = ExcelHandler(file_queue)
    observer = Observer()
    observer.schedule(event_handler, str(SOURCE_DIR), recursive=False)
    observer.start()

    logger.info(f"Watcher started on {SOURCE_DIR} with {NUM_WORKERS} workers and {OUTPUT_WORKERS} output workers")

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

    # Wait for output workers to finish
    for output_worker in output_workers:
        output_worker.join(timeout=5)

    logger.info("Watcher shutdown complete")


if __name__ == "__main__":
    main()
