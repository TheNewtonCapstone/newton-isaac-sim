import logging
import os
import csv


class CSVFileHandler(logging.Handler):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        # Create the file if it doesn't exist and write the header
        with open(self.log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the header if the file is empty
            if os.stat(self.log_path).st_size == 0:
                writer.writerow(['Timestamp', 'Name', 'Level', 'Message'])

    def emit(self, record):
        # Format the log record as a CSV row
        log_entry = self.format(record)
        with open(self.log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)


class DataLogger(logging.Logger):
    def __init__(self, log_path):
        super().__init__(__name__)
        self.log_path = log_path

        # Setup logger during initialization
        self.setup()

    def setup(self):
        # Set the logging level
        self.setLevel(logging.INFO)

        # Create a CSV file handler
        handler = CSVFileHandler(self.log_path)

        # Define the logging format
        formatter = logging.Formatter(
            "%(asctime)s, %(name)s, %(levelname)s, %(message)s"
        )
        handler.setFormatter(formatter)

        # Ensure that the logger doesn't duplicate messages
        if not self.hasHandlers():
            self.addHandler(handler)

# Example usage
if __name__ == "__main__":
    logger = DataLogger('log.csv')
    logger.info("This is an info message.")
    logger.error("This is an error message.")