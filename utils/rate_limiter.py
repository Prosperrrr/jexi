import time
from collections import defaultdict
from datetime import datetime

class RateLimiter:
    """
    Simple rate limiter to prevent spam uploads
    """
    
    def __init__(self, max_requests=5, time_window=60):
        """
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)  # IP -> list of timestamps
    
    def is_allowed(self, identifier):
        """
        Check if a request is allowed
        
        Args:
            identifier: User identifier (IP address, user ID, etc.)
            
        Returns:
            tuple: (allowed: bool, remaining: int, reset_time: int)
        """
        current_time = time.time()
        
        # Get request history for this identifier
        request_times = self.requests[identifier]
        
        # Remove old requests outside the time window
        request_times = [t for t in request_times if current_time - t < self.time_window]
        self.requests[identifier] = request_times
        
        # Check if limit exceeded
        if len(request_times) >= self.max_requests:
            # Calculate when the oldest request will expire
            oldest_request = min(request_times)
            reset_time = int(oldest_request + self.time_window - current_time)
            
            return False, 0, reset_time
        
        # Add current request
        request_times.append(current_time)
        
        remaining = self.max_requests - len(request_times)
        return True, remaining, self.time_window
    
    def reset(self, identifier):
        """Reset rate limit for a specific identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
    
    def get_stats(self):
        """Get current rate limiter statistics"""
        current_time = time.time()
        
        active_users = 0
        total_requests = 0
        
        for identifier, request_times in list(self.requests.items()):
            # Clean up old requests
            request_times = [t for t in request_times if current_time - t < self.time_window]
            
            if request_times:
                active_users += 1
                total_requests += len(request_times)
                self.requests[identifier] = request_times
            else:
                del self.requests[identifier]
        
        return {
            "active_users": active_users,
            "total_recent_requests": total_requests,
            "max_requests_per_window": self.max_requests,
            "time_window_seconds": self.time_window
        }