# Python Data Engineering Complete Guide

> A comprehensive guide to advanced Python concepts essential for data engineers

## Table of Contents
1. [File Operations & Data Handling](#file-operations--data-handling)
2. [Parallel Processing](#parallel-processing)
3. [Decorators](#decorators)
4. [Lambda Functions](#lambda-functions)
5. [List Comprehensions](#list-comprehensions)
6. [Dictionary Comprehensions](#dictionary-comprehensions)
7. [Practice Assignments](#practice-assignments)

---

## File Operations & Data Handling

### Basic File Operations

#### Reading Files with Context Manager (Best Practice)
```python
# Always use 'with' statement - it automatically closes files
with open("data.txt", "r") as file:
    content = file.read()
    
# Reading line by line (memory efficient)
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

#### Writing and Appending Files
```python
# Writing (overwrites existing content)
with open("output.txt", "w") as file:
    file.write("New data")

# Appending (adds to existing content)
with open("output.txt", "a") as file:
    file.write("\nAdditional data")
```

### Handling Large Files Efficiently

For big data files, loading everything into memory can crash your system:

```python
# Reading in chunks (memory-safe)
with open("large_dataset.txt", "r") as file:
    while chunk := file.read(1024):  # Read 1024 bytes at a time
        process_chunk(chunk)

# Processing CSV files in chunks with pandas
import pandas as pd

chunk_size = 10000
for chunk in pd.read_csv("massive_data.csv", chunksize=chunk_size):
    # Process each chunk separately
    processed_chunk = chunk[chunk['sales'] > 1000]
    processed_chunk.to_csv("filtered_data.csv", mode='a', header=False)
```

### Working with CSV and Excel Files

```python
import pandas as pd

# Reading CSV files
df = pd.read_csv("sales_data.csv")
print(df.head())  # First 5 rows
print(df.info())  # Data types and null values

# Writing CSV files
df.to_csv("processed_data.csv", index=False)

# Excel operations
df = pd.read_excel("quarterly_report.xlsx", sheet_name="Q1")
df.to_excel("summary_report.xlsx", sheet_name="Summary", index=False)
```

**Real-world Example: Processing Sales Data**
```python
import pandas as pd

def process_sales_file(filename):
    """Process sales data and create summary report"""
    # Read the data
    df = pd.read_csv(filename)
    
    # Clean and process
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['revenue'] = df['quantity'] * df['price']
    
    # Create summary
    monthly_summary = df.groupby(df['sale_date'].dt.month)['revenue'].sum()
    
    # Save results
    monthly_summary.to_csv("monthly_revenue.csv")
    return monthly_summary
```

---

## Parallel Processing

Parallel processing is crucial for data engineers working with large datasets or multiple data sources.

### Multithreading (I/O-Bound Tasks)

Use threading when your code waits for external resources (file reads, API calls, database queries):

```python
import threading
import requests
import time

def fetch_data_from_api(url, results, index):
    """Fetch data from API endpoint"""
    response = requests.get(url)
    results[index] = response.json()

# Fetch data from multiple APIs simultaneously
urls = [
    "https://api.sales.com/q1",
    "https://api.sales.com/q2", 
    "https://api.sales.com/q3",
    "https://api.sales.com/q4"
]

results = [None] * len(urls)
threads = []

start_time = time.time()

for i, url in enumerate(urls):
    thread = threading.Thread(target=fetch_data_from_api, args=(url, results, i))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print(f"Fetched all data in {time.time() - start_time:.2f} seconds")
```

### Multiprocessing (CPU-Bound Tasks)

Use multiprocessing for heavy computations that can utilize multiple CPU cores:

```python
from multiprocessing import Pool
import pandas as pd

def process_data_chunk(chunk_file):
    """Process a chunk of data - CPU intensive operations"""
    df = pd.read_csv(chunk_file)
    
    # Heavy computation: complex statistical analysis
    df['moving_average'] = df['sales'].rolling(window=30).mean()
    df['volatility'] = df['sales'].rolling(window=30).std()
    
    return df

def parallel_data_processing(file_list):
    """Process multiple data files in parallel"""
    with Pool(processes=4) as pool:
        results = pool.map(process_data_chunk, file_list)
    
    # Combine all results
    final_df = pd.concat(results, ignore_index=True)
    return final_df

# Example usage
data_files = ['sales_2023_q1.csv', 'sales_2023_q2.csv', 
              'sales_2023_q3.csv', 'sales_2023_q4.csv']
combined_data = parallel_data_processing(data_files)
```

### Using concurrent.futures (Recommended Approach)

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd

# For I/O-bound tasks (file operations, API calls)
def download_and_process_file(url):
    """Download and process data file"""
    df = pd.read_csv(url)
    return df.shape[0]  # Return row count

urls = ['https://data.company.com/file1.csv', 
        'https://data.company.com/file2.csv']

with ThreadPoolExecutor(max_workers=4) as executor:
    row_counts = list(executor.map(download_and_process_file, urls))

print(f"Total rows processed: {sum(row_counts)}")

# For CPU-bound tasks (heavy computations)
def calculate_complex_metrics(data_chunk):
    """Perform complex calculations on data chunk"""
    # Simulate heavy computation
    return data_chunk.sum() ** 2

large_dataset = [list(range(1000000)) for _ in range(4)]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(calculate_complex_metrics, large_dataset))
```

---

## Decorators

Decorators are essential for data engineers to add functionality like logging, timing, and error handling without modifying core business logic.

### Basic Decorator Pattern

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

@my_decorator
def process_data(filename):
    print(f"Processing {filename}")
    return f"Processed {filename}"

result = process_data("sales.csv")
```

### Essential Data Engineering Decorators

#### 1. Timing Decorator
```python
import time
import functools

def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_it
def load_large_dataset(filename):
    """Load and process large dataset"""
    import pandas as pd
    df = pd.read_csv(filename)
    return df.shape

# Usage
dataset_info = load_large_dataset("big_data.csv")
# Output: load_large_dataset took 2.45 seconds
```

#### 2. Retry Decorator (For API Calls and Database Connections)
```python
import time
import random

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        print(f"Failed after {max_attempts} attempts")
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def connect_to_database():
    """Simulate unreliable database connection"""
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Database connection failed")
    return "Connected successfully!"

# Usage
try:
    result = connect_to_database()
    print(result)
except ConnectionError as e:
    print(f"Final error: {e}")
```

#### 3. Logging Decorator
```python
import logging
import functools

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Successfully completed {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

@log_execution
def extract_transform_load(source_file, target_file):
    """ETL pipeline function"""
    import pandas as pd
    
    # Extract
    df = pd.read_csv(source_file)
    
    # Transform
    df['processed_date'] = pd.Timestamp.now()
    df['revenue'] = df['quantity'] * df['price']
    
    # Load
    df.to_csv(target_file, index=False)
    return len(df)
```

#### 4. Cache Decorator (For Expensive Operations)
```python
def cache_results(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key in cache:
            print(f"Cache hit for {func.__name__}")
            return cache[key]
        
        print(f"Computing {func.__name__}")
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper

@cache_results
def expensive_calculation(n):
    """Simulate expensive computation"""
    time.sleep(2)  # Simulate delay
    return sum(i**2 for i in range(n))

# First call takes 2 seconds
result1 = expensive_calculation(1000)

# Second call with same parameters is instant
result2 = expensive_calculation(1000)  # Cache hit!
```

---

## Lambda Functions

Lambda functions are perfect for data engineers who need quick, inline functions for data transformations.

### Basic Lambda Syntax

```python
# Regular function
def square(x):
    return x * x

# Lambda equivalent
square = lambda x: x * x

# Multiple arguments
add = lambda x, y: x + y
multiply = lambda x, y, z: x * y * z
```

### Data Processing with Lambda

#### Filtering Data
```python
# Sample employee data
employees = [
    {'name': 'Alice', 'salary': 75000, 'department': 'Engineering'},
    {'name': 'Bob', 'salary': 65000, 'department': 'Sales'},
    {'name': 'Carol', 'salary': 80000, 'department': 'Engineering'},
    {'name': 'David', 'salary': 55000, 'department': 'Marketing'}
]

# Filter high earners (salary > 70000)
high_earners = list(filter(lambda emp: emp['salary'] > 70000, employees))
print("High earners:", [emp['name'] for emp in high_earners])

# Filter by department
engineers = list(filter(lambda emp: emp['department'] == 'Engineering', employees))
print("Engineers:", [emp['name'] for emp in engineers])
```

#### Transforming Data
```python
# Transform salary data
# Add 10% bonus to all salaries
salaries_with_bonus = list(map(lambda emp: {**emp, 'salary_with_bonus': emp['salary'] * 1.1}, employees))

# Extract specific fields
names = list(map(lambda emp: emp['name'], employees))
departments = list(map(lambda emp: emp['department'], employees))

print("Names:", names)
print("Departments:", departments)
```

#### Sorting Data
```python
# Sort employees by different criteria
by_salary = sorted(employees, key=lambda emp: emp['salary'])
by_name = sorted(employees, key=lambda emp: emp['name'])
by_dept_then_salary = sorted(employees, key=lambda emp: (emp['department'], -emp['salary']))

print("Highest paid:", by_salary[-1]['name'])
print("Lowest paid:", by_salary[0]['name'])
```

### Real-world Data Cleaning Example

```python
# Messy sales data
sales_data = [
    " PRODUCT-A ", "product-b", " PRODUCT-C ", "product-d "
]

# Clean product names
cleaned_products = list(map(lambda product: product.strip().upper().replace('-', '_'), sales_data))
print("Cleaned products:", cleaned_products)

# Validate email addresses
emails = ['alice@company.com', 'invalid-email', 'bob@company.com', 'carol@company.com']
valid_emails = list(filter(lambda email: '@' in email and '.' in email, emails))
print("Valid emails:", valid_emails)

# Process transaction amounts
transactions = ['$1,234.56', '$2,345.67', '$345.89']
amounts = list(map(lambda t: float(t.replace('$', '').replace(',', '')), transactions))
print("Transaction amounts:", amounts)
```

---

## List Comprehensions

List comprehensions provide a concise way to create lists and are essential for data transformation tasks.

### Basic List Comprehension Syntax

```python
# Format: [expression for item in iterable if condition]

# Create list of squares
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition - only even squares
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
```

### Data Processing Examples

#### Processing CSV-like Data
```python
# Simulate CSV data rows
csv_rows = [
    "Alice,25,Engineer,75000",
    "Bob,30,Manager,85000", 
    "Carol,28,Developer,70000",
    "David,35,Director,95000"
]

# Convert to dictionaries using list comprehension
employees = [
    {
        'name': row.split(',')[0],
        'age': int(row.split(',')[1]),
        'role': row.split(',')[2],
        'salary': int(row.split(',')[3])
    }
    for row in csv_rows
]

print("Employee data:", employees[0])
```

#### Filtering and Transforming Sales Data
```python
# Daily sales data
daily_sales = [120, 340, 89, 456, 234, 567, 123, 890, 234, 456]

# Sales above $200
high_sales_days = [sale for sale in daily_sales if sale > 200]
print("High sales days:", high_sales_days)

# Add 8% tax to all sales
sales_with_tax = [sale * 1.08 for sale in daily_sales]
print("Sales with tax:", sales_with_tax[:3])  # First 3 values

# Categorize sales performance
performance = ['Excellent' if sale > 400 else 'Good' if sale > 200 else 'Needs Improvement' 
               for sale in daily_sales]
print("Performance categories:", performance[:5])
```

#### Working with Log Data
```python
# Server log entries
log_entries = [
    "2024-01-15 09:30:22 ERROR Database connection failed",
    "2024-01-15 09:31:15 INFO User login successful", 
    "2024-01-15 09:32:45 WARNING High memory usage detected",
    "2024-01-15 09:33:12 ERROR API request timeout",
    "2024-01-15 09:34:33 INFO Data backup completed"
]

# Extract error logs only
error_logs = [log for log in log_entries if 'ERROR' in log]
print("Error logs:", len(error_logs))

# Extract timestamps from all logs
timestamps = [log.split()[0] + ' ' + log.split()[1] for log in log_entries]
print("Timestamps:", timestamps[:2])

# Extract log levels
log_levels = [log.split()[2] for log in log_entries]
print("Log levels:", log_levels)
```

#### Nested List Comprehensions
```python
# Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Flatten matrix
flattened = [num for row in matrix for num in row]
print("Flattened:", flattened)

# Transpose matrix
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print("Transposed:", transposed)

# Filter and transform
# Get even numbers multiplied by 2
even_doubled = [num * 2 for row in matrix for num in row if num % 2 == 0]
print("Even doubled:", even_doubled)
```

---

## Dictionary Comprehensions

Dictionary comprehensions are powerful for creating mappings and transforming data structures.

### Basic Dictionary Comprehension Syntax

```python
# Format: {key_expression: value_expression for item in iterable if condition}

# Create squares dictionary
squares_dict = {x: x**2 for x in range(5)}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From two lists
products = ['laptop', 'mouse', 'keyboard', 'monitor']
prices = [999.99, 29.99, 79.99, 299.99]
product_prices = {product: price for product, price in zip(products, prices)}
print(product_prices)
```

### Data Engineering Applications

#### Configuration Processing
```python
# Environment configuration
config_strings = [
    "DATABASE_HOST=localhost",
    "DATABASE_PORT=5432", 
    "DATABASE_NAME=sales_db",
    "API_KEY=abc123xyz",
    "DEBUG_MODE=true"
]

# Convert to configuration dictionary
config = {
    item.split('=')[0]: item.split('=')[1] 
    for item in config_strings
}
print("Configuration:", config)

# Type conversion for specific keys
typed_config = {
    key: (int(value) if key.endswith('_PORT') 
          else bool(value) if key.endswith('_MODE') 
          else value)
    for key, value in config.items()
}
print("Typed config:", typed_config)
```

#### Sales Data Analysis
```python
# Sales data by region
regions = ['North', 'South', 'East', 'West']
quarterly_sales = [
    [1200000, 1350000, 1180000, 1420000],  # Q1
    [1350000, 1280000, 1390000, 1510000],  # Q2
    [1180000, 1420000, 1250000, 1380000],  # Q3
    [1520000, 1380000, 1460000, 1650000]   # Q4
]

# Create sales summary by region
regional_totals = {
    region: sum(quarterly_sales[quarter][i] for quarter in range(4))
    for i, region in enumerate(regions)
}
print("Regional totals:", regional_totals)

# Top performing regions (sales > 5,000,000)
top_regions = {
    region: total for region, total in regional_totals.items() 
    if total > 5000000
}
print("Top regions:", top_regions)
```

#### Employee Data Processing
```python
# Employee records
employees = [
    {'id': 101, 'name': 'Alice Johnson', 'dept': 'Engineering', 'salary': 85000},
    {'id': 102, 'name': 'Bob Smith', 'dept': 'Marketing', 'salary': 65000},
    {'id': 103, 'name': 'Carol Davis', 'dept': 'Engineering', 'salary': 92000},
    {'id': 104, 'name': 'David Wilson', 'dept': 'Sales', 'salary': 58000}
]

# Create ID to employee mapping
id_to_employee = {emp['id']: emp for emp in employees}

# Create department salary mapping (engineering only)
eng_salaries = {
    emp['name']: emp['salary'] 
    for emp in employees 
    if emp['dept'] == 'Engineering'
}
print("Engineering salaries:", eng_salaries)

# Salary grades
salary_grades = {
    emp['name']: (
        'Senior' if emp['salary'] > 80000 
        else 'Mid' if emp['salary'] > 60000 
        else 'Junior'
    )
    for emp in employees
}
print("Salary grades:", salary_grades)
```

#### Data Cleaning and Transformation
```python
# Messy customer data
raw_customer_data = {
    ' ALICE JOHNSON ': ' alice@email.com ',
    'bob smith': 'BOB@EMAIL.COM',
    ' Carol Davis': 'carol@email.com ',
    'DAVID WILSON ': ' david@email.com'
}

# Clean and normalize data
clean_customers = {
    name.strip().title(): email.strip().lower()
    for name, email in raw_customer_data.items()
}
print("Clean customers:", clean_customers)

# Create reverse mapping (email to name)
email_to_name = {email: name for name, email in clean_customers.items()}
print("Email to name mapping:", email_to_name)
```

---

## Practice Assignments

### Assignment 1: File Handling
**Objective**: Create a CSV processor that filters and transforms sales data.

```python
def process_sales_data(input_file, output_file, min_amount=100):
    """
    Read sales CSV, filter records with amount > min_amount,
    add calculated fields, and save to new file.
    """
    import pandas as pd
    
    # Your implementation here
    # 1. Read CSV file
    # 2. Filter rows where 'amount' > min_amount
    # 3. Add 'tax' column (8% of amount)
    # 4. Add 'total' column (amount + tax)
    # 5. Save to output file
    pass

# Test data structure:
# product,quantity,price,amount
# Laptop,2,500,1000
# Mouse,10,25,250
# Keyboard,5,80,400
```

### Assignment 2: Parallel Processing
**Objective**: Process multiple data files simultaneously.

```python
def parallel_file_processor(file_list):
    """
    Process multiple CSV files in parallel using ThreadPoolExecutor.
    Each file should be read, processed, and summary statistics calculated.
    """
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    
    def process_single_file(filename):
        # Your implementation here
        # 1. Read the file
        # 2. Calculate summary stats (mean, sum, count)
        # 3. Return results
        pass
    
    # Your parallel processing implementation here
    pass
```

### Assignment 3: Advanced Decorators
**Objective**: Create a comprehensive decorator for data pipeline functions.

```python
def data_pipeline_decorator(log_performance=True, handle_errors=True, max_retries=3):
    """
    Create a decorator that:
    1. Logs function execution time
    2. Handles and logs errors
    3. Implements retry logic
    4. Validates input/output data
    """
    def decorator(func):
        # Your implementation here
        pass
    return decorator

@data_pipeline_decorator(log_performance=True, max_retries=2)
def extract_transform_load(source_file, target_file):
    """Example ETL function to be decorated"""
    import pandas as pd
    
    # Extract
    df = pd.read_csv(source_file)
    
    # Transform
    df['processed_at'] = pd.Timestamp.now()
    
    # Load
    df.to_csv(target_file, index=False)
    
    return len(df)
```

### Assignment 4: Lambda and List Comprehensions Challenge
**Objective**: Process complex nested data using lambda functions and comprehensions.

```python
# Sample data: nested structure representing sales by region and product
sales_data = {
    'North': {
        'laptops': [1200, 1500, 900, 1800],
        'mice': [25, 30, 20, 35],
        'keyboards': [80, 75, 90, 85]
    },
    'South': {
        'laptops': [1100, 1400, 1000, 1600],
        'mice': [20, 25, 30, 28],
        'keyboards': [70, 80, 75, 82]
    }
}

# Challenges:
# 1. Use list comprehension to flatten all sales amounts
# 2. Use lambda with filter to find products with average sales > 100
# 3. Create a dictionary comprehension for total sales by region
# 4. Use lambda with map to apply 10% discount to all amounts

# Your solutions here:
```

### Assignment 5: Real-world Data Pipeline
**Objective**: Build a complete data processing pipeline combining all concepts.

```python
class DataPipeline:
    """
    A complete data pipeline that demonstrates:
    - File operations with error handling
    - Parallel processing for multiple data sources
    - Decorators for logging and monitoring
    - Lambda functions for data transformations
    - Comprehensions for data filtering and mapping
    """
    
    def __init__(self, config):
        self.config = config
        self.processed_files = []
    
    @time_it
    @log_execution
    @retry(max_attempts=3)
    def extract_data(self, source_files):
        """Extract data from multiple sources in parallel"""
        # Your implementation here
        pass
    
    def transform_data(self, raw_data):
        """Transform data using comprehensions and lambda functions"""
        # Your implementation here
        pass
    
    def load_data(self, transformed_data, target_file):
        """Load processed data to target destination"""
        # Your implementation here
        pass
    
    def run_pipeline(self, source_files, target_file):
        """Execute the complete pipeline"""
        # Your implementation here
        pass

# Usage example:
# pipeline = DataPipeline(config={'chunk_size': 10000, 'parallel_workers': 4})
# pipeline.run_pipeline(['data1.csv', 'data2.csv'], 'processed_output.csv')
```

## Key Takeaways for Data Engineers

1. **File Operations**: Always use context managers (`with` statement) and process large files in chunks
2. **Parallel Processing**: Use threading for I/O-bound tasks, multiprocessing for CPU-bound tasks
3. **Decorators**: Essential for cross-cutting concerns like logging, timing, and error handling
4. **Lambda Functions**: Perfect for quick transformations in data processing pipelines
5. **Comprehensions**: More efficient and readable than traditional loops for data transformations
6. **Error Handling**: Always implement proper error handling and retry mechanisms for production systems

## Additional Resources

- **Performance**: Use `cProfile` to identify bottlenecks in your data processing code
- **Memory Management**: Consider using generators for very large datasets
- **Testing**: Write unit tests for your data processing functions
- **Documentation**: Use docstrings to document your data pipeline functions
- **Version Control**: Track changes to your data processing scripts with Git

---

*This guide covers essential Python concepts for data engineers. Practice these concepts with real datasets to build proficiency in data processing and pipeline development.*
