import os
import json
import logging
import time
from datetime import datetime, timedelta
import sqlite3
import threading
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsService:
    def __init__(self):
        """Initialize the metrics service with database connection"""
        self.db_path = os.path.join(os.getcwd(), 'app', 'data', 'metrics.db')
        self.db_lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for storing metrics"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create requests table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    response_time REAL,
                    status_code INTEGER,
                    error TEXT
                )
                ''')
                
                # Create model_metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    input_size INTEGER,
                    batch_size INTEGER
                )
                ''')
                
                conn.commit()
                conn.close()
                
                logger.info("Metrics database initialized")
        except Exception as e:
            logger.error(f"Error initializing metrics database: {str(e)}")
    
    def log_request(self, endpoint, response_time, status_code=200, error=None):
        """
        Log a request to the database
        
        Args:
            endpoint (str): The API endpoint that was called
            response_time (float): The response time in seconds
            status_code (int): The HTTP status code
            error (str, optional): Error message if any
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO requests (timestamp, endpoint, response_time, status_code, error) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), endpoint, response_time, status_code, error)
                )
                
                conn.commit()
                conn.close()
            
            # Log errors separately for easier tracking
            if error:
                self._log_error(endpoint, error, status_code)
                
        except Exception as e:
            logger.error(f"Error logging request: {str(e)}")
    
    def _log_error(self, endpoint, error, status_code):
        """
        Log detailed error information
        
        Args:
            endpoint (str): The API endpoint that was called
            error (str): Error message
            status_code (int): The HTTP status code
        """
        try:
            error_log_path = os.path.join(os.getcwd(), 'logs', 'errors.log')
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            
            with open(error_log_path, 'a') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} | {endpoint} | {status_code} | {error}\n")
                
        except Exception as e:
            logger.error(f"Error logging detailed error: {str(e)}")
    
    def get_performance_metrics(self, start_date=None, end_date=None, interval='daily'):
        """
        Get performance metrics for the specified time range
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            interval (str, optional): Aggregation interval (hourly, daily, weekly, monthly)
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                # Default to last 7 days if not specified
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Convert to datetime objects
            start_datetime = datetime.strptime(f"{start_date} 00:00:00", '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", '%Y-%m-%d %H:%M:%S')
            
            # Determine SQL date format based on interval
            if interval == 'hourly':
                date_format = "%Y-%m-%d %H:00:00"
                group_by = "strftime('%Y-%m-%d %H:00:00', timestamp)"
            elif interval == 'daily':
                date_format = "%Y-%m-%d"
                group_by = "strftime('%Y-%m-%d', timestamp)"
            elif interval == 'weekly':
                date_format = "%Y-%W"
                group_by = "strftime('%Y-%W', timestamp)"
            elif interval == 'monthly':
                date_format = "%Y-%m"
                group_by = "strftime('%Y-%m', timestamp)"
            else:
                date_format = "%Y-%m-%d"
                group_by = "strftime('%Y-%m-%d', timestamp)"
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get request counts by time period and status code
                cursor.execute(f"""
                SELECT 
                    {group_by} as period,
                    status_code,
                    COUNT(*) as count,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time
                FROM requests
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY period, status_code
                ORDER BY period
                """, (start_datetime.isoformat(), end_datetime.isoformat()))
                
                rows = cursor.fetchall()
                
                # Get endpoint performance
                cursor.execute(f"""
                SELECT 
                    endpoint,
                    COUNT(*) as count,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time
                FROM requests
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY endpoint
                ORDER BY count DESC
                """, (start_datetime.isoformat(), end_datetime.isoformat()))
                
                endpoint_rows = cursor.fetchall()
                
                conn.close()
            
            # Process the data
            metrics = {
                'time_series': defaultdict(list),
                'endpoints': [],
                'summary': {
                    'total_requests': 0,
                    'error_rate': 0,
                    'avg_response_time': 0
                }
            }
            
            # Process time series data
            total_requests = 0
            error_requests = 0
            response_times = []
            
            for row in rows:
                period, status_code, count, avg_time, min_time, max_time = row
                
                metrics['time_series'][period].append({
                    'status_code': status_code,
                    'count': count,
                    'avg_response_time': avg_time
                })
                
                total_requests += count
                if status_code >= 400:
                    error_requests += count
                
                if avg_time:
                    response_times.extend([avg_time] * count)
            
            # Process endpoint data
            for row in endpoint_rows:
                endpoint, count, avg_time, min_time, max_time = row
                
                metrics['endpoints'].append({
                    'endpoint': endpoint,
                    'count': count,
                    'avg_response_time': avg_time,
                    'min_response_time': min_time,
                    'max_response_time': max_time
                })
            
            # Calculate summary
            metrics['summary']['total_requests'] = total_requests
            metrics['summary']['error_rate'] = (error_requests / total_requests) * 100 if total_requests > 0 else 0
            metrics['summary']['avg_response_time'] = np.mean(response_times) if response_times else 0
            
            # Convert defaultdict to regular dict for JSON serialization
            metrics['time_series'] = dict(metrics['time_series'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'error': str(e)}
    
    def get_error_logs(self, start_date=None, end_date=None, error_type=None, limit=100):
        """
        Get error logs for the specified filters
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            error_type (str, optional): Filter by error type
            limit (int, optional): Maximum number of logs to return
            
        Returns:
            list: Error logs
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                # Default to last 7 days if not specified
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Convert to datetime objects
            start_datetime = datetime.strptime(f"{start_date} 00:00:00", '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", '%Y-%m-%d %H:%M:%S')
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query based on filters
                query = """
                SELECT 
                    id, timestamp, endpoint, response_time, status_code, error
                FROM requests
                WHERE timestamp BETWEEN ? AND ? AND error IS NOT NULL
                """
                params = [start_datetime.isoformat(), end_datetime.isoformat()]
                
                if error_type:
                    query += " AND error LIKE ?"
                    params.append(f"%{error_type}%")
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                conn.close()
            
            # Process the data
            error_logs = []
            for row in rows:
                id, timestamp, endpoint, response_time, status_code, error = row
                
                error_logs.append({
                    'id': id,
                    'timestamp': timestamp,
                    'endpoint': endpoint,
                    'response_time': response_time,
                    'status_code': status_code,
                    'error': error
                })
            
            return error_logs
            
        except Exception as e:
            logger.error(f"Error getting error logs: {str(e)}")
            return {'error': str(e)}
    
    def get_metrics_summary(self):
        """
        Get a summary of key metrics
        
        Returns:
            dict: Metrics summary
        """
        try:
            # Get current date and time
            now = datetime.now()
            today = now.strftime('%Y-%m-%d')
            yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            last_week = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Convert to datetime objects
            today_start = datetime.strptime(f"{today} 00:00:00", '%Y-%m-%d %H:%M:%S')
            yesterday_start = datetime.strptime(f"{yesterday} 00:00:00", '%Y-%m-%d %H:%M:%S')
            last_week_start = datetime.strptime(f"{last_week} 00:00:00", '%Y-%m-%d %H:%M:%S')
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get today's metrics
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_requests,
                    AVG(response_time) as avg_response_time
                FROM requests
                WHERE timestamp >= ?
                """, (today_start.isoformat(),))
                
                today_row = cursor.fetchone()
                
                # Get yesterday's metrics
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_requests,
                    AVG(response_time) as avg_response_time
                FROM requests
                WHERE timestamp >= ? AND timestamp < ?
                """, (yesterday_start.isoformat(), today_start.isoformat()))
                
                yesterday_row = cursor.fetchone()
                
                # Get last week's metrics
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_requests,
                    AVG(response_time) as avg_response_time
                FROM requests
                WHERE timestamp >= ? AND timestamp < ?
                """, (last_week_start.isoformat(), yesterday_start.isoformat()))
                
                last_week_row = cursor.fetchone()
                
                # Get top endpoints
                cursor.execute("""
                SELECT 
                    endpoint,
                    COUNT(*) as count,
                    AVG(response_time) as avg_response_time
                FROM requests
                WHERE timestamp >= ?
                GROUP BY endpoint
                ORDER BY count DESC
                LIMIT 5
                """, (last_week_start.isoformat(),))
                
                top_endpoints_rows = cursor.fetchall()
                
                # Get recent errors
                cursor.execute("""
                SELECT 
                    timestamp, endpoint, status_code, error
                FROM requests
                WHERE timestamp >= ? AND error IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 5
                """, (last_week_start.isoformat(),))
                
                recent_errors_rows = cursor.fetchall()
                
                conn.close()
            
            # Process the data
            today_total, today_errors, today_avg_time = today_row
            yesterday_total, yesterday_errors, yesterday_avg_time = yesterday_row
            last_week_total, last_week_errors, last_week_avg_time = last_week_row
            
            # Calculate metrics
            today_error_rate = (today_errors / today_total) * 100 if today_total > 0 else 0
            yesterday_error_rate = (yesterday_errors / yesterday_total) * 100 if yesterday_total > 0 else 0
            
            # Calculate trends
            request_trend = self._calculate_trend(today_total, yesterday_total)
            error_rate_trend = self._calculate_trend(today_error_rate, yesterday_error_rate, lower_is_better=True)
            response_time_trend = self._calculate_trend(today_avg_time, yesterday_avg_time, lower_is_better=True)
            
            # Process top endpoints
            top_endpoints = []
            for row in top_endpoints_rows:
                endpoint, count, avg_time = row
                top_endpoints.append({
                    'endpoint': endpoint,
                    'count': count,
                    'avg_response_time': avg_time
                })
            
            # Process recent errors
            recent_errors = []
            for row in recent_errors_rows:
                timestamp, endpoint, status_code, error = row
                recent_errors.append({
                    'timestamp': timestamp,
                    'endpoint': endpoint,
                    'status_code': status_code,
                    'error': error
                })
            
            # Build summary
            summary = {
                'current': {
                    'total_requests': today_total,
                    'error_rate': today_error_rate,
                    'avg_response_time': today_avg_time
                },
                'previous': {
                    'total_requests': yesterday_total,
                    'error_rate': yesterday_error_rate,
                    'avg_response_time': yesterday_avg_time
                },
                'trends': {
                    'requests': request_trend,
                    'error_rate': error_rate_trend,
                    'response_time': response_time_trend
                },
                'top_endpoints': top_endpoints,
                'recent_errors': recent_errors
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_trend(self, current, previous, lower_is_better=False):
        """
        Calculate trend percentage and direction
        
        Args:
            current (float): Current value
            previous (float): Previous value
            lower_is_better (bool): Whether lower values are better
            
        Returns:
            dict: Trend information
        """
        if previous is None or previous == 0:
            return {
                'percentage': 0,
                'direction': 'stable'
            }
        
        percentage = ((current - previous) / previous) * 100
        
        if percentage > 1:
            direction = 'up' if not lower_is_better else 'down'
        elif percentage < -1:
            direction = 'down' if not lower_is_better else 'up'
        else:
            direction = 'stable'
        
        return {
            'percentage': abs(percentage),
            'direction': direction
        }
    
    def log_model_metric(self, model_id, metric_name, metric_value, input_size=None, batch_size=None):
        """
        Log a model metric to the database
        
        Args:
            model_id (str): The model identifier
            metric_name (str): Name of the metric (e.g., 'accuracy', 'loss', 'inference_time')
            metric_value (float): Value of the metric
            input_size (int, optional): Size of the input data
            batch_size (int, optional): Batch size used
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO model_metrics (timestamp, model_id, metric_name, metric_value, input_size, batch_size) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), model_id, metric_name, metric_value, input_size, batch_size)
                )
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error logging model metric: {str(e)}")
    
    def get_model_metrics(self, model_id=None, metric_name=None, start_date=None, end_date=None):
        """
        Get model metrics for the specified filters
        
        Args:
            model_id (str, optional): Filter by model ID
            metric_name (str, optional): Filter by metric name
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            dict: Model metrics
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                # Default to last 30 days if not specified
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Convert to datetime objects
            start_datetime = datetime.strptime(f"{start_date} 00:00:00", '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", '%Y-%m-%d %H:%M:%S')
            
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query based on filters
                query = """
                SELECT 
                    timestamp, model_id, metric_name, metric_value, input_size, batch_size
                FROM model_metrics
                WHERE timestamp BETWEEN ? AND ?
                """
                params = [start_datetime.isoformat(), end_datetime.isoformat()]
                
                if model_id:
                    query += " AND model_id = ?"
                    params.append(model_id)
                
                if metric_name:
                    query += " AND metric_name = ?"
                    params.append(metric_name)
                
                query += " ORDER BY timestamp"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                conn.close()
            
            # Process the data
            metrics = defaultdict(lambda: defaultdict(list))
            
            for row in rows:
                timestamp, model_id, metric_name, metric_value, input_size, batch_size = row
                
                metrics[model_id][metric_name].append({
                    'timestamp': timestamp,
                    'value': metric_value,
                    'input_size': input_size,
                    'batch_size': batch_size
                })
            
            # Convert defaultdict to regular dict for JSON serialization
            return json.loads(json.dumps(metrics))
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return {'error': str(e)} 